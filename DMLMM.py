import torch
import numpy as np 
from scipy import special 
from scipy import stats
from sklearn.decomposition import SparsePCA
from sklearn.cluster import KMeans
from statistics import mean 

class DMLMM:
    def __init__(self,y,x,K,D):
        self.y,self.x,self.K,self.D = y,x,K,D
        self.dtype = torch.float64 
        self.eps=1e-6 #some small value
        self.EPS=1e20 #some large value
        self.L=len(self.K)
        # Hyperparmeters
        self.G=0.5*np.ones(self.L) 
        self.nu=1*np.ones(self.L)
        self.A=2.5*np.ones(self.L)
        self.A0=1
        self.rho=[0.5*np.ones(i) for i in self.K]
        self.n=len(self.y)
        self.D[0]=len(self.x[0][0])
        #Initialization step 
        self.mu_m,self.mu_sigma,self.b_m,self.b_sigma,self.delta_a,self.delta_b,self.psi_a,self.psi_b,self.g_a,self.g_b,self.h_a,self.h_b,self.c_a,self.c_b,self.tau_a,self.tau_b,self.xi_a,self.xi_b,self.p_d=([] for i in range(19)) 
        
        init=min(self.n,8192) #The initalization is done on a smaller subsample to decrease runtime on large datasets
        w=self.n/init #weight to correct for subsample
        Z=[]
        eps_loc=[]
        var_starter=[]
        for i in range(self.n):
            Z.append(np.matmul(np.matmul(np.linalg.pinv(np.matmul(self.x[i].T,self.x[i])+.1*np.diag(np.ones(D[0]))),self.x[i].T),self.y[i]))#
            eps_loc.append(((np.matmul(self.x[i],Z[-1])-self.y[i])**2).mean())
            var_starter.append(np.var((np.matmul(self.x[i],Z[-1])-self.y[i]).numpy()))
        Z=torch.stack(Z)  
        eps_loc=np.sum(eps_loc)
        self.sigma_a=torch.tensor((self.n+1)/2,dtype=self.dtype)
        self.sigma_b=torch.tensor(((self.n+1)/2+1)*np.quantile(var_starter,q=0.5),dtype=self.dtype)
        self.s_b=torch.tensor(1/eps_loc+1/self.A0**2,dtype=self.dtype)
        self.s_a=torch.tensor(1,dtype=self.dtype)
        for l in range(self.L):
            Z_new=[]    
            mu_m_loc=torch.empty(self.K[l],self.D[l],dtype=self.dtype)
            mu_sigma_loc=torch.empty(self.K[l],self.D[l],dtype=self.dtype)
            g_b_loc=torch.empty(self.K[l],self.D[l],dtype=self.dtype)
            b_m_loc=torch.empty(self.K[l],self.D[l]*self.D[l+1],dtype=self.dtype)
            h_b_loc=torch.empty(self.K[l],self.D[l]*self.D[l+1],dtype=self.dtype)
            p_d_loc=torch.empty(self.K[l],dtype=self.dtype) 
            delta_a_loc=torch.empty(self.K[l],self.D[l],dtype=self.dtype)
            delta_b_loc=torch.empty(self.K[l],self.D[l],dtype=self.dtype)
            psi_b_loc=torch.empty(self.K[l],self.D[l],dtype=self.dtype)

            labels = KMeans(n_clusters=self.K[l],n_init='auto').fit(Z).labels_
            for k in range(self.K[l]):
                p_d_loc[k]=np.count_nonzero(labels == k)*w
                if (labels==k).sum()>D[l+1]:
                    subset=Z[labels== k] 
                else: #add random elements when cluster contains less then D[l+1] elements
                    hilfselement=(labels== k)
                    hilfselement[np.random.randint(low=0,high=len(hilfselement),size=self.D[l+1])]=True
                    subset=Z[hilfselement] 
                subset_mean=subset.mean(-2)
                subset_centered=subset-subset_mean
                mu_m_loc[k]=subset_mean.detach().clone().unsqueeze(0)
                mu_sigma_loc[k]=subset_centered.pow(2).mean(0).detach().clone().unsqueeze(0)
                g_b_loc[k]=0.5*(subset_mean.detach().clone().unsqueeze(0).pow(2)/self.G[l]+1)
                subset_pca=np.asarray(subset_centered)
                pca = SparsePCA(n_components=self.D[l+1])
                pca.fit(subset_pca)
                Z_new.append(pca.fit_transform(subset))
                B_nr=np.matmul(np.linalg.pinv(np.matmul(np.transpose(pca.components_),pca.components_)),np.transpose(pca.components_))
                b_m_loc[k]=torch.tensor(B_nr.flatten(),dtype=self.dtype)
                h_b_loc[k]=b_m_loc[k].flatten().clone().unsqueeze(0).pow(2)*0.5*self.nu[l]**(-2)+0.5  
                eps_loc=(subset_centered-torch.mm(b_m_loc[k].reshape(self.D[l],self.D[l+1]),torch.tensor(Z_new[-1]).t()).t()).clamp(min=-1,max=1)
                delta_a_loc[k]=(p_d_loc[k]+1)/2
                delta_b_loc[k]=w*eps_loc.pow(2).sum(0)/2
                psi_b_loc[k]=w*eps_loc.pow(2).mean(0).pow(-1)+1/self.A[l]**2
            self.mu_m.append(mu_m_loc)
            self.mu_sigma.append(w/self.n*mu_sigma_loc)
            self.g_b.append(g_b_loc)
            self.b_m.append(b_m_loc)
            self.h_b.append(h_b_loc)
            self.p_d.append(p_d_loc)
            self.b_sigma.append(0.001*torch.ones(self.K[l],self.D[l]*self.D[l+1],dtype=self.dtype))
            self.h_a.append(torch.ones(self.K[l],self.D[l]*self.D[l+1],dtype=self.dtype))
            self.c_a.append(torch.ones(self.K[l],self.D[l]*self.D[l+1],dtype=self.dtype))
            self.c_b.append(torch.ones(self.K[l],self.D[l]*self.D[l+1],dtype=self.dtype))
            self.xi_a.append(torch.ones(self.K[l],dtype=self.dtype))
            self.xi_b.append(self.D[l]*self.D[l+1]+2+1/self.nu[l]**2*torch.ones(self.K[l],dtype=self.dtype))
            self.tau_a.append(self.D[l]*self.D[l+1]/2*torch.ones(self.K[l],dtype=self.dtype))
            self.tau_b.append(3/2*self.nu[l]**2*torch.ones(self.K[l],dtype=self.dtype))  
            self.delta_a.append(delta_a_loc)
            self.delta_b.append(delta_b_loc)
            self.psi_a.append(torch.ones(self.K[l],self.D[l],dtype=self.dtype))
            self.psi_b.append(psi_b_loc)
            self.g_a.append(torch.ones(self.K[l],self.D[l],dtype=self.dtype))
            Z=torch.from_numpy(np.concatenate(Z_new))
        #Avoid overflow errors
        for par in [self.mu_sigma,self.b_sigma,self.delta_a,self.delta_b,self.psi_a,self.psi_b,self.g_a,self.g_b,self.h_a,self.h_b,self.c_a,self.c_b,self.tau_a,self.tau_b,self.xi_a,self.xi_b,self.p_d]:
            for l in range(self.L):
                  par[l].clamp_(min=self.eps,max=self.EPS)
        for par in [self.sigma_a,self.sigma_b,self.s_a,self.s_b]:
            par.clamp_(min=self.eps,max=self.EPS)
        for par in [self.mu_m,self.b_m]:
            for l in range(self.L):
                par[l].clamp_(min=-self.EPS,max=self.EPS)   
    
    def build_estimators(self):
    # This method returns the parameters of the GMM corresponding corresponding to the posterior mode based on the current estimates for lambda of the DGMM
        delta_hat=[]
        p_hat=[]
        mu_hat=[]
        b_hat=[]
        for l in range(self.L):
            delta_hat_loc=[]
            b_hat_loc=[]
            for k in range(self.K[l]):
                delta_hat_loc.append((self.delta_b[l][k]*(self.delta_a[l][k]+1).pow(-1)).detach())
                b_hat_loc.append(self.b_m[l][k].reshape(self.D[l],self.D[l+1]).detach())
            mu_hat.append(self.mu_m[l].detach())
            p_hat.append((self.p_d[l]/self.p_d[l].sum()).detach())
            delta_hat.append(delta_hat_loc)
            b_hat.append(b_hat_loc)
        mu_gmm_hat=[]
        sigma_gmm_hat=[]
        weight_gmm_hat=[]
        paths=self.all_paths(self.K)
        for path in paths:
            weight=1
            for l in range(self.L):
                weight*=p_hat[l][path[l]]
            weight_gmm_hat.append(weight)
            m, S = self.comp_par(path,mu_hat,delta_hat,b_hat,self.D,self.L)
            mu_gmm_hat.append(m)
            sigma_gmm_hat.append(S)
        paths=self.all_paths(self.K)
        weight_gmm_hat=torch.stack(weight_gmm_hat)
        mu_gmm_hat = torch.stack(mu_gmm_hat)
        sigma_gmm_hat = torch.stack(sigma_gmm_hat)
        return paths, weight_gmm_hat, mu_gmm_hat, sigma_gmm_hat
    
    def all_paths(self,K):
    # This method takes a list K as argument. K contains the number of clusters in each layer of a DGMM. The method returns a list of all possible paths through the DGMM.    
        paths=list()
        anzahl_paths=1
        for k in K:
            anzahl_paths*=k
        k=K[0]
        remaining_options=anzahl_paths/k
        c=0
        for s in range(anzahl_paths):
            if s%remaining_options==0:
                c+=1
            paths.append([c-1])
        for l in range(1,len(K)):
            k=K[l]
            remaining_options*=1/k
            c=0
            for s in range(anzahl_paths):
                if s%remaining_options==0:
                    c+=1
                paths[s].append(c%k)
        return paths
    
    def comp_par(self,path,mu,delta,B,D,L):
    # This methods returns the mean and the variance parameter for the marginal distirbution of z up to a given layer along a given path
        m=torch.zeros(D[L]).to(torch.float64)
        S=torch.diag(torch.ones(D[L])).to(torch.float64)
        for l_ in range(len(path)):
            l=L-1-l_
            k=path[-(l_+1)]
            m=mu[l][k]+torch.matmul(B[l][k],m)
            S=torch.matmul(B[l][k],torch.matmul(S,B[l][k].T))+torch.diag(delta[l][k])
        return m, S
    
    def calc_elbo(self,index,z_m,z_sigma,alpha):
    # This method calculates the ELBO and returns its value using a minibatch at a given point.   
        batchsize=len(index)
        T=0
        T11=-0.5*self.D[self.L]*np.log(2*np.pi)-0.5*(z_m[self.L].pow(2)+z_sigma[self.L].pow(2)).sum()
        for i in range(batchsize):
            T11+=-0.5*len(self.y[index[i]])*(np.log(2*np.pi)+self.sigma_b.log()-self.sigma_a.digamma())-0.5*self.sigma_a*self.sigma_b.pow(-1)*(self.y[index[i]]-torch.matmul(self.x[index[i]],z_m[0][i])).pow(2).sum()+torch.matmul(self.x[index[i]].pow(2),z_sigma[0][i].pow(2)).sum()
            for l in range(self.L):
                for k in range(self.K[l]):
                    var=torch.matmul(self.b_m[l][k].reshape(self.D[l],self.D[l+1]).pow(2),z_sigma[l+1][i])+torch.matmul(self.b_sigma[l][k].reshape(self.D[l],self.D[l+1]),(z_m[l+1][i].pow(2)))+torch.matmul(self.b_sigma[l][k].reshape(self.D[l],self.D[l+1]),z_sigma[l+1][i])
                    T11+=alpha[l][i][k]*(-0.5*self.D[l]*np.log(2*np.pi)-0.5*(self.delta_b[l][k].log()-self.delta_a[l][k].digamma()+self.delta_a[l][k]*self.delta_b[l][k].pow(-1)*((z_m[l][i]-self.mu_m[l][k]-torch.matmul(self.b_m[l][k].reshape(self.D[l],self.D[l+1]),z_m[l+1][i])).pow(2)+z_sigma[l][i]+self.mu_sigma[l][k]+var)).sum())
        T+=self.n/batchsize*T11
        
        T+=-0.5*(self.s_b.log()-self.s_a.digamma())-np.log(special.gamma(0.5))-1.5*(self.sigma_b.log()-self.sigma_a.digamma())-self.sigma_a*self.s_a*self.sigma_b.pow(-1)*self.s_b.pow(-1)
        T+=-0.5*np.log(self.A0**2)-np.log(special.gamma(0.5))-1.5*(self.s_b.log()-self.s_a.digamma())-self.s_a*self.s_b.pow(-1)*1/self.A0**2
        T+=self.entropy_inversegamma(self.sigma_a,self.sigma_b)
        T+=self.entropy_inversegamma(self.s_a,self.s_b)
        
        for l in range(self.L):
           T12=(-0.5*np.log(2*np.pi*self.D[l])-0.5*(self.g_b[l].log()-self.g_a[l].digamma())-0.5*1/self.G[l]*self.g_a[l]/self.g_b[l]*(self.mu_sigma[l]+self.mu_m[l].pow(2))).sum()
           T13=0 
           for k in range(self.K[l]):
                T13+=(-0.5*np.log(2*np.pi)+0.5*(self.h_a[l][k].digamma()-self.h_b[l][k].log())-0.5*(self.tau_b[l][k].log()-self.tau_a[l][k].digamma())-0.5*self.tau_a[l][k]*self.h_a[l][k]*self.tau_b[l][k].pow(-1)*self.h_b[l][k].pow(-1)*(self.b_sigma[l][k]+self.b_m[l][k].pow(2))).sum()
           T14=(-0.5*(self.psi_b[l].log()-self.psi_a[l].digamma())-np.log(special.gamma(0.5))-1.5*(self.delta_b[l].log()-self.delta_a[l].digamma())-self.delta_a[l]*self.psi_a[l]*self.delta_b[l].pow(-1)*self.psi_b[l].pow(-1)).sum()
           T15=(-0.5*np.log(self.A[l]**2)-np.log(special.gamma(0.5))-1.5*(self.psi_b[l].log()-self.psi_a[l].digamma())-self.psi_a[l]*self.psi_b[l].pow(-1)*1/self.A[l]**2).sum()
           T16=(0.5*np.log(0.5)-np.log(special.gamma(0.5))-1.5*(self.g_b[l].log()-self.g_a[l].digamma())-0.5*self.g_a[l]*self.g_b[l].pow(-1)).sum()
           T17=(-np.log(special.gamma(0.5))+0.5*(self.c_a[l].digamma()-self.c_b[l].log())-0.5*(self.h_a[l].digamma()-self.h_b[l].log())-self.c_a[l]*self.h_a[l]*self.c_b[l].pow(-1)*self.h_b[l].pow(-1)).sum()
           T18=-np.log(special.gamma(self.rho[l].sum()))
           for k in range(self.K[l]):
               T18+=np.log(special.gamma(self.rho[l][k]))+(self.rho[l][k]-1)*(self.p_d[l][k].digamma()-self.p_d[l].sum().digamma())
           T19=0
           for i in range(batchsize):
               for k in range(self.K[l]):
                   T19+=alpha[l][i][k]*(self.p_d[l][k].digamma()-self.p_d[l].sum().digamma())
           T19*=self.n/batchsize
           T110=(-np.log(special.gamma(0.5))-0.5*(self.c_a[l].digamma()-self.c_b[l].log())-self.c_a[l]*self.c_b[l].pow(-1)).sum()
           T111=(-np.log(special.gamma(0.5))-0.5*(self.xi_b[l].log()-self.xi_a[l].digamma())-1.5*(self.tau_b[l].log()-self.tau_a[l].digamma())-self.tau_a[l]*self.xi_a[l]*self.tau_b[l].pow(-1)*self.xi_b[l].pow(-1)).sum()
           T112=(-np.log(special.gamma(0.5))-0.5*np.log(self.nu[l]**2)-1.5*(self.xi_b[l].log()-self.xi_a[l].digamma())-1/self.nu[l]**2*self.xi_a[l]*self.xi_b[l].pow(-1)).sum()
           T21=self.entropy_normal(self.mu_sigma[l])
           T22=self.entropy_normal(self.b_sigma[l])
           T23=self.entropy_normal(z_sigma[l])*self.n/batchsize
           T24=self.entropy_dirichlet(self.p_d[l])
           T25=self.entropy_inversegamma(self.delta_a[l],self.delta_b[l])
           T26=self.entropy_inversegamma(self.psi_a[l],self.psi_b[l])
           T27=self.entropy_inversegamma(self.g_a[l],self.g_b[l])
           T28=self.entropy_gamma(self.h_a[l],self.h_b[l])
           T29=(alpha[l]*alpha[l].log()).sum()*self.n/batchsize
           T210=self.entropy_gamma(self.c_a[l],self.c_b[l])
           T211=self.entropy_inversegamma(self.tau_a[l],self.tau_b[l])
           T212=self.entropy_inversegamma(self.xi_a[l],self.xi_b[l])
           T1=T12+T13+T14+T15+T16+T17+T18+T19+T110+T111+T112 
           T2=T21+T22+T23+T24+T25+T26+T27+T28+T29+T210+T211+T212
           T+=T1-T2
        T-=self.entropy_normal(z_sigma[self.L])*self.n/batchsize
        return T

    def entropy_normal(self,sigma):
    # Calculates the negative entropy for indipendend normal distirbutions. It takes the variances as argument
        ent=-0.5*((2*np.pi*np.e*sigma).log()).sum()
        return ent

    def entropy_gamma(self,a,b):
    # Calculates the negative entropy for indipendend gamma distirbutions. It takes the parameters of the distributions as argument
        ent=(-a+b.log()-a.lgamma()-(1-a)*a.digamma()).sum()
        return ent


    def entropy_inversegamma(self,a,b):
    # Calculates the negative entropy for indipendend inversegamma distirbutions. It takes the parameters of the distributions as argument
        ent=((1+a)*a.digamma()-a-b.log()-a.lgamma()).sum()
        return ent

    def entropy_dirichlet(self,a):
    # Calculates the negative entropy for indipendend Dirichlet distirbutions. It takes the parameters of the distributions as argument
        a_0=a.sum()
        return -a.lgamma().sum()+a.sum().lgamma()-(a_0-len(a))*a_0.digamma()+((a-1)*a.digamma()).sum()
    
    def train(self,max_iter,threshold):
        batchsize=int(min(max(0.05*self.n,2*self.K[0]),1024))
        warm_up_ranganath = 10
        # dummys for natuaral gradients
        mu_m_ng,mu_sigma_ng,b_m_ng,b_sigma_ng,delta_a_ng,delta_b_ng,psi_a_ng,psi_b_ng,g_a_ng,g_b_ng,h_a_ng,h_b_ng,c_a_ng,c_b_ng,tau_a_ng,tau_b_ng,xi_a_ng,xi_b_ng,p_d_ng=([] for i in range(19))    
        for l in range(self.L):
            mu_m_ng.append(torch.zeros(self.K[l],self.D[l],requires_grad=False,dtype=self.dtype))
            mu_sigma_ng.append(torch.zeros(self.K[l],self.D[l],requires_grad=False,dtype=self.dtype))
            b_m_ng.append(torch.zeros(self.K[l],self.D[l]*self.D[l+1],requires_grad=False,dtype=self.dtype))
            b_sigma_ng.append(torch.zeros(self.K[l],self.D[l]*self.D[l+1], requires_grad=False,dtype=self.dtype))
            p_d_ng.append(torch.zeros(self.K[l],requires_grad=False,dtype=self.dtype))
            delta_a_ng.append(torch.zeros(self.K[l],self.D[l], requires_grad=False,dtype=self.dtype))
            delta_b_ng.append(torch.zeros(self.K[l],self.D[l], requires_grad=False,dtype=self.dtype))
            psi_a_ng.append(torch.zeros(self.K[l],self.D[l], requires_grad=False,dtype=self.dtype))
            psi_b_ng.append(torch.zeros(self.K[l],self.D[l], requires_grad=False,dtype=self.dtype))
            g_a_ng.append(torch.zeros(self.K[l],self.D[l], requires_grad=False,dtype=self.dtype))
            g_b_ng.append(torch.zeros(self.K[l],self.D[l],requires_grad=False,dtype=self.dtype))
            h_a_ng.append(torch.zeros(self.K[l],self.D[l]*self.D[l+1], requires_grad=False,dtype=self.dtype))
            h_b_ng.append(torch.zeros(self.K[l],self.D[l]*self.D[l+1],requires_grad=False,dtype=self.dtype))
            c_a_ng.append(torch.zeros(self.K[l],self.D[l]*self.D[l+1], requires_grad=False,dtype=self.dtype))
            c_b_ng.append(torch.zeros(self.K[l],self.D[l]*self.D[l+1], requires_grad=False,dtype=self.dtype))
            tau_a_ng.append(torch.zeros(self.K[l], requires_grad=False,dtype=self.dtype))
            tau_b_ng.append(torch.zeros(self.K[l],requires_grad=False,dtype=self.dtype))    
            xi_a_ng.append(torch.zeros(self.K[l], requires_grad=False,dtype=self.dtype))
            xi_b_ng.append(torch.zeros(self.K[l], requires_grad=False,dtype=self.dtype))  
        sigma_a_ng=torch.tensor(0,requires_grad=False,dtype=self.dtype)
        sigma_b_ng=torch.tensor(0,requires_grad=False,dtype=self.dtype)
        s_a_ng=torch.tensor(0,requires_grad=False,dtype=self.dtype)
        s_b_ng=torch.tensor(0,requires_grad=False,dtype=self.dtype)
        #### Starting point of the gradient ascent algorithm #####
        for par in [self.mu_m,self.mu_sigma,self.b_m,self.b_sigma,self.delta_a,self.delta_b,self.psi_a,self.psi_b,self.g_a,self.g_b,self.h_a,self.h_b,self.c_a,self.c_b,self.tau_a,self.tau_b,self.xi_a,self.xi_b,self.p_d]:
            for l in range(self.L):
                par[l].requires_grad=True
        for par in [self.sigma_a,self.sigma_b,self.s_a,self.s_b]:
            par.requires_grad=True
        elbo_ts=[]
        for wdh_global in range(max_iter): 
            #draw minibatch
            index=np.random.randint(low=0,high=self.n,size=batchsize)
            #update local parameters
            with torch.no_grad():
                eps_alpha=self.eps
                alpha=[]
                z_m=[]
                z_sigma=[]
                for l in range(self.L):
                    alpha.append(eps_alpha*torch.ones(batchsize,self.K[l],requires_grad=False,dtype=self.dtype))
                for l in range(self.L+1):
                    z_m.append(torch.rand(batchsize,self.D[l],requires_grad=False,dtype=self.dtype))
                    z_sigma.append(torch.rand(batchsize,self.D[l],requires_grad=False,dtype=self.dtype))
                paths, weight_gmm_hat, mu_gmm_hat, sigma_gmm_hat = self.build_estimators()
                delta_hat=[]
                mu_hat=[]
                b_hat=[]
                for l in range(self.L):
                    delta_hat_loc=[]
                    b_hat_loc=[]
                    for k in range(self.K[l]):
                        delta_hat_loc.append((self.delta_b[l][k]*(self.delta_a[l][k]+1).pow(-1)).detach())
                        b_hat_loc.append(self.b_m[l][k].reshape(self.D[l],self.D[l+1]).detach())
                    mu_hat.append(self.mu_m[l].detach())
                    delta_hat.append(delta_hat_loc)
                    b_hat.append(b_hat_loc)
                for i in range(batchsize):
                    log_likelihood=-torch.inf
                    opt_path=paths[np.random.randint(0,len(paths)-1)]
                    opt_c=0
                    for c in range(len(paths)): 
                        try:
                            log_likelihood_proposed=torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.matmul(self.x[index[i]],mu_gmm_hat[c]),covariance_matrix=torch.matmul(torch.matmul(self.x[index[i]],sigma_gmm_hat[c]),self.x[index[i]].T)+self.sigma_b/(self.sigma_a+1)*torch.diag(torch.ones(len(self.y[index[i]])))).log_prob(self.y[index[i]])+np.log(weight_gmm_hat[c])
                        except:
                            log_likelihood_proposed=-np.inf
                        if (log_likelihood_proposed>=log_likelihood):
                            opt_path=paths[c]
                            opt_c=c
                            log_likelihood=log_likelihood_proposed
                    xtex=torch.matmul(torch.matmul(self.x[index[i]].T,(self.sigma_a+1)/self.sigma_b*torch.diag(torch.ones(len(self.y[index[i]]),dtype=self.dtype))),self.x[index[i]])
                    xtey=torch.matmul(torch.matmul(self.x[index[i]].T,torch.linalg.pinv(self.sigma_b/(self.sigma_a+1)*torch.diag(torch.ones(len(self.y[index[i]]),dtype=self.dtype)))),self.y[index[i]])
                    V=torch.pinverse(xtex+torch.pinverse(sigma_gmm_hat[opt_c]))
                    m=torch.matmul(V,xtey+torch.matmul(torch.pinverse(sigma_gmm_hat[opt_c]),mu_gmm_hat[opt_c]))
                    z_m[0][i]=m
                    z_sigma[0][i]=torch.diag(V)
                    fix_z=m
                    for l in range(self.L):
                        k=opt_path[l]
                        alpha[l][i][k]=1-(self.K[l]-1)*eps_alpha
                        path_rest=opt_path[l+1:]
                        if path_rest:
                            m, S = self.comp_par(path_rest,mu_hat,delta_hat,b_hat,self.D,self.L)
                        else:
                            m=torch.zeros(self.D[self.L]).to(torch.float64)
                            S=torch.diag(torch.ones(self.D[self.L])).to(torch.float64)
                        delta_inv=torch.diag(self.delta_b[l][k].pow(-1)*(self.delta_a[l][k]+1))
                        covariance_inverse=torch.mm(torch.mm(self.b_m[l][k].reshape(self.D[l],self.D[l+1]).t(),delta_inv),self.b_m[l][k].reshape(self.D[l],self.D[l+1]))+torch.pinverse(S)
                        covariance=torch.pinverse(covariance_inverse)
                        z_sigma[l+1][i]=torch.diag(covariance)
                        z_m[l+1][i]=torch.mm(covariance,torch.mm(self.b_m[l][k].reshape(self.D[l],self.D[l+1]).t(),torch.mm(delta_inv,(fix_z-self.mu_m[l][k]).unsqueeze(1)))-torch.mm(torch.pinverse(S),m.unsqueeze(1))).t()[0]
                        fix_z=z_m[l+1][i]         
            #calc gradients
            elbo=self.calc_elbo(index,z_m,z_sigma,alpha)
            elbo_ts.append(elbo.detach().item())
            elbo.backward()
            with torch.no_grad():
                mu=[self.mu_m,self.b_m]
                mu_ng=[mu_m_ng,b_m_ng]
                sigma=[self.mu_sigma,self.b_sigma]
                sigma_ng=[mu_sigma_ng,b_sigma_ng]
                zipped=zip(mu,mu_ng,sigma,sigma_ng)
                for (mu,mu_ng,sigma,sigma_ng) in zipped:
                    for l in range(self.L):
                        mu_ng[l].data=sigma[l]*mu[l].grad
                        sigma_ng[l].data=2*sigma[l].pow(2)*sigma[l].grad
                #natural gradients for (inverse)gamma parameters
                a=[self.delta_a,self.psi_a,self.g_a,self.h_a,self.c_a,self.tau_a,self.xi_a]
                a_ng=[delta_a_ng,psi_a_ng,g_a_ng,h_a_ng,c_a_ng,tau_a_ng,xi_a_ng]
                b=[self.delta_b,self.psi_b,self.g_b,self.h_b,self.c_b,self.tau_b,self.xi_b]
                b_ng=[delta_b_ng,psi_b_ng,g_b_ng,h_b_ng,c_b_ng,tau_b_ng,xi_b_ng]
                zipped=zip(a,a_ng,b,b_ng)
                for (a,a_ng,b,b_ng) in zipped:
                    for l in range(self.L):
                        det=(a[l].polygamma(1)*a[l]-1).pow(-1)*b[l].pow(2)
                        a_ng[l].data=det*(a[l]*b[l].pow(-2)*a[l].grad+b[l].pow(-1)*b[l].grad)
                        b_ng[l].data=det*(b[l].pow(-1)*a[l].grad+a[l].polygamma(1)*b[l].grad)
                a=[self.sigma_a,self.s_a]
                a_ng=[sigma_a_ng,s_a_ng]
                b=[self.sigma_b,self.s_b]
                b_ng=[sigma_b_ng,s_b_ng]
                zipped=zip(a,a_ng,b,b_ng)
                for (a,a_ng,b,b_ng) in zipped:
                    det=(a.polygamma(1)*a-1).pow(-1)*b.pow(2)
                    a_ng.data=det*(a*b.pow(-2)*a.grad+b.pow(-1)*b.grad)
                    b_ng.data=det*(b.pow(-1)*a.grad+a.polygamma(1)*b.grad)
                #natural gradients for dirichlet parameters
                for l in range(self.L):
                    fim=torch.diag(self.p_d[l].polygamma(1))-(self.p_d[l].sum().polygamma(1)*torch.ones(len(self.p_d[l]),len(self.p_d[l])))
                    fim_inv=torch.pinverse(fim)
                    p_d_ng[l].data=torch.matmul(fim_inv,self.p_d[l].grad)
                #update parameters
                #update stepsize
                grad=self.build_ranganath_gradient(mu_m_ng,mu_sigma_ng,b_m_ng,b_sigma_ng,delta_a_ng,delta_b_ng,psi_a_ng,psi_b_ng,g_a_ng,g_b_ng,h_a_ng,h_b_ng,c_a_ng,c_b_ng,tau_a_ng,tau_b_ng,xi_a_ng,xi_b_ng,p_d_ng,self.s_a,self.s_b,self.sigma_a,self.sigma_b)
                if wdh_global<warm_up_ranganath:
                    if wdh_global==0:
                        tau_ranganath=warm_up_ranganath
                        g_ranganath=1/warm_up_ranganath*grad
                        h_ranganath=1/warm_up_ranganath*np.square(grad).sum()
                    else:
                        g_ranganath+=1/warm_up_ranganath*grad
                        h_ranganath+=1/warm_up_ranganath*np.square(grad).sum()
                    stepsize=0
                else:
                    g_ranganath=(1-1/tau_ranganath)*g_ranganath+1/tau_ranganath*grad
                    h_ranganath=(1-1/tau_ranganath)*h_ranganath+1/tau_ranganath*np.square(grad).sum()
                    stepsize=min(0.98,1/h_ranganath*np.square(g_ranganath).sum())
                    tau_ranganath=(1-stepsize)*tau_ranganath+1
                parameters=[self.mu_m,self.mu_sigma,self.b_m,self.b_sigma,self.delta_a,self.delta_b,self.psi_a,self.psi_b,self.g_a,self.g_b,self.h_a,self.h_b,self.c_a,self.c_b,self.tau_a,self.tau_b,self.xi_a,self.xi_b,self.p_d]
                nat_grad=[mu_m_ng,mu_sigma_ng,b_m_ng,b_sigma_ng,delta_a_ng,delta_b_ng,psi_a_ng,psi_b_ng,g_a_ng,g_b_ng,h_a_ng,h_b_ng,c_a_ng,c_b_ng,tau_a_ng,tau_b_ng,xi_a_ng,xi_b_ng,p_d_ng]
                zipped=zip(parameters,nat_grad)
                #update parameters
                for (par,nat_grad) in zipped:
                    with torch.no_grad():
                        for l in range(self.L):
                            par[l].add_(stepsize*nat_grad[l])
                            par[l].grad.zero_()      
                parameters=[self.sigma_a,self.sigma_b,self.s_a,self.s_b]
                nat_grad=[sigma_a_ng,sigma_b_ng,s_a_ng,s_b_ng]
                zipped=zip(parameters,nat_grad)
                #update parameters
                for (par,nat_grad) in zipped:
                    with torch.no_grad():
                        par.add_(stepsize*nat_grad)
                        par.grad.zero_()    
                for par in [self.mu_sigma,self.b_sigma,self.delta_a,self.delta_b,self.psi_a,self.psi_b,self.g_a,self.g_b,self.h_a,self.h_b,self.c_a,self.c_b,self.tau_a,self.tau_b,self.xi_a,self.xi_b,self.p_d]:
                    with torch.no_grad():
                        for l in range(self.L):
                              par[l].clamp_(min=self.eps,max=self.EPS)
                for par in [self.mu_m,self.b_m]:
                    with torch.no_grad():
                        for l in range(self.L):
                            par[l].clamp_(min=-self.EPS,max=self.EPS)
                for par in [self.sigma_a,self.sigma_b,self.s_a,self.s_b]:
                    with torch.no_grad():
                        par.clamp_(min=self.eps,max=self.EPS)
            print(wdh_global)
            #check stopping condition
            if abs(mean(elbo_ts[-round(len(elbo_ts)*0.05):])-mean(elbo_ts[-round(len(elbo_ts)*0.1):]))<=threshold and wdh_global>250:
                break
                  
        for par in [self.mu_m,self.mu_sigma,self.b_m,self.b_sigma,self.delta_a,self.delta_b,self.psi_a,self.psi_b,self.g_a,self.g_b,self.h_a,self.h_b,self.c_a,self.c_b,self.tau_a,self.tau_b,self.xi_a,self.xi_b,self.p_d]:
            for l in range(self.L):
                par[l].requires_grad=False
        for par in [self.sigma_a,self.sigma_b,self.s_a,self.s_b]:
            par.requires_grad=False
            
        self.cluster_gmm()

    def build_ranganath_gradient(self,mu_m_ng,mu_sigma_ng,b_m_ng,b_sigma_ng,delta_a_ng,delta_b_ng,psi_a_ng,psi_b_ng,g_a_ng,g_b_ng,h_a_ng,h_b_ng,c_a_ng,c_b_ng,tau_a_ng,tau_b_ng,xi_a_ng,xi_b_ng,p_d_ng,s_a_ng,s_b_ng,sigma_a_ng,sigma_b_ng):
        new_g=[]
        for grad in [mu_m_ng,b_m_ng,b_sigma_ng,delta_a_ng,delta_b_ng,psi_a_ng,psi_b_ng,g_a_ng,g_b_ng,h_a_ng,h_b_ng,c_a_ng,c_b_ng,tau_a_ng,tau_b_ng,xi_a_ng,xi_b_ng,p_d_ng]:
            for subgrad in grad:
                new_g.append(subgrad.numpy().flatten())
        new_g=np.concatenate( new_g, axis=0 )
        for subgrad in [s_a_ng,s_b_ng,sigma_a_ng,sigma_b_ng]:
            new_g=np.append(new_g,subgrad.numpy())
        return new_g
    
    def cluster_gmm(self):
        paths, w, m, s = self.build_estimators()
        sigma2_hat = self.sigma_b/(self.sigma_a+1)
        cluster = []
        for i in range(len(self.y)):
            log_likelihood=-torch.inf
            for c in range(len(w)): 
                log_likelihood_proposed=torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.matmul(self.x[i],m[c]),covariance_matrix=torch.matmul(torch.matmul(self.x[i],s[c]),self.x[i].T)+sigma2_hat*torch.diag(torch.ones(len(self.y[i])))).log_prob(self.y[i])+np.log(w[c])
                if (log_likelihood_proposed>=log_likelihood):
                    opt_c=c
                    log_likelihood=log_likelihood_proposed
            cluster.append(opt_c)
        cluster_used = np.unique(cluster)
        for c in range(len(w)):
            if c not in cluster_used:
                w[c]=0
        w = w/w.sum()
        self.s_gmm = s[w>0]
        self.m_gmm = m[w>0,:]
        self.w_gmm = w[w>0]

    def sample_gmm(self,w,m,s,n):
        sample=[]
        anz=np.random.multinomial(n,w)
        for c in range(len(w)):
            sample.append(np.random.multivariate_normal(mean=m[c],cov=s[c],tol=1e-6,size=anz[c],check_valid='ignore'))
        sample = np.vstack(sample)
        np.random.shuffle(sample)
        return sample
    
    def predict_y(self,design_pred,y_obs,n):
        w_pred = []
        m_pred = []
        s_pred = []
        for k in range(len(self.w_gmm)):
            w_pred.append(self.w_gmm[k])
            m_pred.append(np.dot(design_pred,self.m_gmm[k]))
            s_pred.append(np.dot((np.dot(design_pred,self.s_gmm[k])),design_pred.T)+(self.sigma_b/(self.sigma_a+1)).item()*np.identity(len(design_pred)))
        d = len(y_obs)
        w_post = []
        m_post = []
        s_post = []
        for k in range(len(w_pred)):
            mu_theta = m_pred[k][d:]
            mu_y = m_pred[k][0:d]
            sigma_y = s_pred[k][0:d,0:d]
            sigma_theta = s_pred[k][d:,d:]
            sigma_ytheta = s_pred[k][0:d,d:]
            sigma_thetay = s_pred[k][d:,0:d]
            w_post.append(w_pred[k]*stats.multivariate_normal.pdf(y_obs,mean=mu_y,cov=sigma_y,allow_singular=True))
            m_post.append(mu_theta+np.matmul(np.matmul(sigma_thetay,np.linalg.pinv(sigma_y)),(y_obs-mu_y)))
            s_post.append(sigma_theta-np.matmul(np.matmul(sigma_thetay,np.linalg.pinv(sigma_y)),sigma_ytheta))
        w_post=np.asarray(w_post)/np.sum(w_post)
        if n==0:
            return w_post,m_post,s_post
        else:
            return self.sample_gmm(w_post,m_post,s_post,n)
