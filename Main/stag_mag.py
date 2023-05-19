import numpy as np

def coord_to_site(Ly,x,y):
    return Ly*x+y

def generate_sublattices_square(Lx,Ly):
    A_coords = []
    B_coords = []
    A_sites = []
    B_sites = []
    
    for nx in range(Lx):
        for ny in range(Ly):
            if nx%2 == 0:
                if ny%2 ==0:
                    A_coords.append((nx,ny))
                    A_sites.append(coord_to_site(Lx,nx,ny))
                else:
                    B_coords.append((nx,ny))
                    B_sites.append(coord_to_site(Lx,nx,ny))
            else:
                if ny%2 ==0:
                    B_coords.append((nx,ny))
                    B_sites.append(coord_to_site(Lx,nx,ny))
                else:
                    A_coords.append((nx,ny))
                    A_sites.append(coord_to_site(Lx,nx,ny))

    return A_coords,B_coords,A_sites,B_sites


def calculate_stag_mag(Lx,Ly,samples):
    # generate sublattices
    a,b,asites,bsites = generate_sublattices_square(Lx,Ly)
    # do it for spin 1/2 units
    samples = 0.5*(2*samples - 1)
    # calculate sublattice A magnetization
    magA = np.sum(samples[:,asites],axis=1)
    # calculate sublattice B magnetization
    magB = np.sum(samples[:,bsites],axis=1)
    # calculate staggered magnetization
    stag_mags = (-1*magA + magB)/(Lx*Ly)
    avg_stag_mag = np.mean(stag_mags)
    abs_avg_stag_mag = np.mean(abs(stag_mags))
    var = np.var(abs(stag_mags))
    return stag_mags,avg_stag_mag,abs_avg_stag_mag,var

