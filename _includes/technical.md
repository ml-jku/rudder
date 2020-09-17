A finite Markov decision process (MDP) $$\mathcal{P}$$ is a 6-tuple $$\mathcal{P}=(\mathcal{S},\mathcal{A},\mathcal{R},p,\gamma)$$:

* finite sets $$\mathcal{S}$$  of states $$s$$ (random variable $$S_{t}$$ at time $$t$$)
* finite sets $$\mathcal{A}$$ of actions $$a$$ (random variable $$A_{t}$$ at time $$t$$)
* finite sets $$\mathcal{R}$$ of rewards $$r$$ (random variable $$R_{t}$$ at time $$t$$)
* transition-reward distribution $$ p=(S_{t+1}=s',R_{t+1}=r \vert S_t=s,A_t=a) $$
* discount factor $$\gamma$$  

The expected reward is the sum over all transition-reward distributions:   

$$r(s,a)=\sum_r rp(r~\vert~s,a) . $$

The return is for a finite horizon MDP with sequence length $$T$$ and $$\gamma=1$$ is given by  

$$G_t = \sum_{k=0}^{T-t} R_{t+k+1}$$

The action-value function $$q^{\pi}(s,a)$$ for policy $$\pi = p(A_{t+1}=a'~ \vert ~S_{t+1}=s')$$ is   

$$q^{\pi}(s,a) = E_{\pi}[G_t~\vert~S_t=s, A_t=a]$$

Goal of learning is to maximize the expected return at time t=0, that is  

$$
v_0^{\pi}=E_{\pi}[G_0] \ .
$$

