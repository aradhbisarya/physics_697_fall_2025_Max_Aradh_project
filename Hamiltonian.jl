using ITensors, ITensorMPS

m = 1
w = 1
N = 10
F = 1
C = 3

function l(n, c)
    return ((n-1)*F*C) + c - 1
end

let 
    sites = siteinds("S=1/2", (N*F*C))

    #Mass
    Mass = AutoMPO()
    for n=1: N 
        i = isodd(n) ? -1 : 1
        for c=1: C
            Mass += 0.5*m*i, "Z", l(n, c) + 1
        end
    end

    #Hopping
    Hopping = AutoMPO()
    for n=1: N-1
        for c=1: C
            for i=1 : 2
                s1 = l(n+1, c)
                s2 = l(n, c)
                if i==1
                    Hopping += "S+", s1 + 1
                    for k=s1 : s2 - 1
                        Hopping += -1, "Z", k + 1
                    end
                     Hopping += "S-", s2 + 1
                else
                    Hopping += "S+", s2 + 1
                    for k=s2 : s1 - 1
                        Hopping += -1, "Z", k + 1
                    end
                     Hopping += "S-", s1 + 1
                end
            end
        end
    end

    H = MPO(Hopping + Mass, sites)

    nsweeps = 5 # number of sweeps is 5
    maxdim = [10,20,100,100,200] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error

    psi0 = random_mps(sites;linkdims=2)

    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    print(energy)

end