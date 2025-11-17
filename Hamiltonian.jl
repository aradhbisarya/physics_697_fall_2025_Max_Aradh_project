using ITensors, ITensorMPS
using Plots

m = 2
w = 1
J = 1
N = 3
F = 1
C = 3


function l(n, c)
    return ((n-1)*F*C) + c - 1
end

function addOp(arr, op, site)
    push!(arr, op)
    push!(arr, site)
end

function Q_22_n_Q_22_m(coeff,n,m)
    H = AutoMPO()
    H += coeff/16, "Sz", l(n, 1) + 1, "Sz", l(m, 1) + 1
    H += -coeff/16, "Sz", l(n, 2) + 1, "Sz", l(m, 1) + 1
    H += -coeff/16, "Sz", l(n, 1) + 1, "Sz", l(m, 2) + 1
    H += coeff/16, "Sz", l(n, 2) + 1, "Sz", l(m, 2) + 1
    return H
end


function Q_33_n_Q_33_m(coeff,n,m)
    H = AutoMPO()
    H += coeff/48, "Sz", l(n, 1) + 1, "Sz", l(m, 1) + 1
    H += coeff/48, "Sz", l(n, 1) + 1, "Sz", l(m, 2) + 1
    H += -coeff/24, "Sz", l(n, 1) + 1, "Sz", l(m, 3) + 1

    H += coeff/48, "Sz", l(n, 2) + 1, "Sz", l(m, 1) + 1
    H += coeff/48, "Sz", l(n, 2) + 1, "Sz", l(m, 2) + 1
    H += -coeff/24, "Sz", l(n, 2) + 1, "Sz", l(m, 3) + 1

    H += -coeff/24, "Sz", l(n, 3) + 1, "Sz", l(m, 1) + 1
    H += -coeff/24, "Sz", l(n, 3) + 1, "Sz", l(m, 2) + 1
    H += coeff/12, "Sz", l(n, 3) + 1, "Sz", l(m, 3) + 1
    return H
end

function Qd_21_n_Q_21_m(coeff,n,m)
    H = AutoMPO()
    H += coeff/2, "S+", l(n, 1) + 1, "Sz", l(n, 1) + 1,"Sz", l(n, 2) + 1, "S-", l(n, 2) + 1, "S+", l(m, 2) + 1, "Sz", l(m, 2) + 1,"Sz", l(m, 1) + 1, "S-", l(m, 1) + 1
    return H
end

function Q_21_n_Qd_21_m(coeff,n,m)
    H = AutoMPO()
    H += coeff/2, "S+", l(n, 2) + 1, "Sz", l(n, 2) + 1,"Sz", l(n, 1) + 1, "S-", l(n, 1) + 1, "S+", l(m, 1) + 1, "Sz", l(m, 1) + 1,"Sz", l(m, 2) + 1, "S-", l(m, 2) + 1
    return H
end


function Qd_31_n_Q_31_m(coeff,n,m)
    H = AutoMPO()
    H += coeff/2, "S+", l(n, 1) + 1, "Sz", l(n, 1) + 1, "Sz", l(n, 2) + 1,"Sz", l(n, 3) + 1, "S-", l(n, 3) + 1, "S+", l(m, 3) + 1, "Sz", l(m, 3) + 1,"Sz", l(m, 2) + 1,"Sz", l(m, 1) + 1, "S-", l(m, 1) + 1
    return H
end

function Q_31_n_Qd_31_m(coeff,n,m)
    H = AutoMPO()
    H += coeff/2, "S+", l(n, 3) + 1, "Sz", l(n, 3) + 1, "Sz", l(n, 2) + 1,"Sz", l(n, 1) + 1, "S-", l(n, 1) + 1, "S+", l(m, 1) + 1, "Sz", l(m, 1) + 1,"Sz", l(m, 2) + 1,"Sz", l(m, 3) + 1, "S-", l(m, 3) + 1
    return H
end


function Qd_32_n_Q_32_m(coeff,n,m)
    H = AutoMPO()
    H += coeff/2, "S+", l(n, 2) + 1, "Sz", l(n, 2) + 1,"Sz", l(n, 3) + 1, "S-", l(n, 3) + 1, "S+", l(m, 3) + 1, "Sz", l(m, 3) + 1,"Sz", l(m, 2) + 1, "S-", l(m, 2) + 1
    return H
end

function Q_32_n_Qd_32_m(coeff,n,m)
    H = AutoMPO()
    H += coeff/2, "S+", l(n, 1) + 1, "Sz", l(n, 1) + 1,"Sz", l(n, 2) + 1, "S-", l(n, 2) + 1, "S+", l(m, 2) + 1, "Sz", l(m, 2) + 1,"Sz", l(m, 1) + 1, "S-", l(m, 1) + 1
    return H
end

function Q_n_m(coeff,n,m)
    H = AutoMPO()

    H += Q_22_n_Q_22_m(coeff,n,m)
    H += Q_33_n_Q_33_m(coeff,n,m)

    H += Qd_31_n_Q_31_m(coeff,n,m)
    H += Q_31_n_Qd_31_m(coeff,n,m)
    
    H += Qd_32_n_Q_32_m(coeff,n,m)
    H += Q_32_n_Qd_32_m(coeff,n,m)
    
    H += Qd_21_n_Q_21_m(coeff,n,m)
    H += Q_21_n_Qd_21_m(coeff,n,m)
    return H
end

function number_operator(psi)
    
    Num = AutoMPO()
    for n=1:N
        for c=1:C
            Num += w/2, "Sz", l(2*n, c) + 1
            Num += -w/2, "Sz", l(2*n-1, c) + 1
        end
    end
    # Calculate the expectation value of the operator with respect to the MPS
    expectation_val = inner(psi', toMPO(Num), psi)
    # Print the result
    println("Expectation value: ", expectation_val)
    return expectation_val
end

let 
    #Mass
    Mass = AutoMPO()
    for n=1: N 
        i = isodd(n) ? -1 : 1
        coeff = 0.5*m*i
        for c=1: C
            Mass += coeff, "Z", l(n, c) + 1
            Mass += coeff, "Id", l(n, c) + 1
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
                    temp = []
                    coeff = 1
                    addOp(temp, "S+", s1 + 1)
                    for k=s2 : s1 - 1
                        coeff *= -1
                        addOp(temp, "Z", k + 1)
                    end
                    addOp(temp, "S-", s2 + 1)
                    pushfirst!(temp, coeff)
                    Hopping += tuple(temp...)
                else
                    temp = []
                    coeff = 1
                    addOp(temp, "S+", s2 + 1)
                    for k=s2 : s1 - 1
                        coeff *= -1
                        addOp(temp, "Z", k + 1)
                    end
                    addOp(temp, "S-", s1 + 1)
                    pushfirst!(temp, coeff)
                    Hopping += tuple(temp...)
                end
            end
        end
    end

    #Electric
    Electric = AutoMPO()
    for n=1: N-1
        println("Adding Q($n, $n)")
        Electric += Q_n_m(J*(N*F*C-n),n,n)
    end

    for n=1: N-2
        for m=n+1: N-1
            println("Adding Q($n, $m)")
            Electric += Q_n_m(2*J*(N*F*C-m),n,m)
    
        end
    end


    sites = siteinds("S=1/2", N*F*C)
    psi0 = random_mps(sites)

    H = MPO(Hopping + Mass + Electric  , sites)
        
    nsweeps = 5 # number of sweeps is 5
    maxdim = [10,20,100,100,200] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error

    psi0 = random_mps(sites;linkdims=2)

    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)

    print(energy)

    number_operator(psi)

end