#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  plot.py
#  
#  Copyright 2024 Luis A Martinez Tossas
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import time
import scipy.optimize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import pandas as pd
import netCDF4 as ncdf
from scipy.interpolate import interp1d
from scipy import integrate

from scipy.ndimage import gaussian_filter1d

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.gridspec import GridSpec

from scipy.stats import linregress

matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 2.5
matplotlib.rcParams['axes.labelsize'] = 16

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def main():
    '''
    Main program
    '''
    
    # ~ figure_3()
    # ~ figure_4()
    figure_5()
    # ~ figure_6_8()
    # ~ figure_7()

def figure_3():
    '''
    Figure of FLLT for different epsilon/c
    '''
    
    epsilon_chord_ls = np.logspace(np.log10(0.1), np.log10(5), 40)
    
    eps_dr = 10 # The converged ratio eps/dr (should be > 5 for convergence)
    
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca() 
    
    cs = ['blue', 'green', 'orange', 'c', 'k',]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('new_cmap', cs,)

    for ic, epsilon_chord in enumerate(epsilon_chord_ls):

        N = np.round(eps_dr * 12.5 / epsilon_chord).astype(int)

        for equation_uy in ['G',]:    
            actuator = actuatorClass(
                        N=N, 
                        chord=1,
                        L=12.5,
                        twist=6,
                        epsilon_chord=epsilon_chord,
                        U=1,
                        lookup_table='NACA64_A17.dat',
                        case_name='Constant_Chord',
                        equation_uy=equation_uy,
                        )            
                        
            # The label for the legend
            label = '$\epsilon/c =$' + "{:.2f}".format(epsilon_chord)
    
            # Solve the filtered lifting line theory equations        
            actuator.solve()
    
            print('mean phi', actuator.phi.mean())
            print('mean uy',  actuator.uy.mean())
    
    
            # The colormap inteprolated
            c = cmap( (ic+1)/ len(epsilon_chord_ls))
            # ~ c = colors[im]
            # ~ c = colors[ic]
    
            if equation_uy=='G': 
                ls='-'
            else:
                ls='--'
            # Plot the induced velocity
            ax.plot(actuator.rad / actuator.L, - actuator.uy, ls,
                        c=c,
                        lw=3,
                        label=label)
            
    ax.grid(True)

    ax.set_xlabel(r'$z/S$')
    ax.set_ylabel(r"$u_y'/U_\infty$")
    
    norm = matplotlib.colors.BoundaryNorm(epsilon_chord_ls, cmap.N)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, ticks=[.1, 0.25, 0.5, 1, 2, 5])
    cbar.set_label(r'$\epsilon/c$', )#rotation=0)

    plt.savefig('figure_3.pdf', bbox_inches='tight')   

def figure_4():
    '''
    FLLT plots for difference epsilon and resolutions
    '''
    # Inputs to change the figure 

    # The list of epsilon/chord
    # ~ epsilon_chord_ls = [.15, .25, .5, 1, 2, 4]
    epsilon_chord_ls = [.25, .5, 1, 2]
    eps_dr = np.array([ .5, .55, .6, .65, .7, .75, .8, 1, 1.5, 2, 4, 10, 20,]) #[::-1] # grid resolution (eps/dr)

    ms = np.linspace(1, 6, len(eps_dr))[::-1]  # marker size

    fig, axs = plt.subplots(nrows=2, ncols=len(epsilon_chord_ls)//2, 
        sharex=True,
        sharey=True,
        figsize=(12, 4))
    
    '''
    Loop through different number of grid points
    '''

    markers = ['o', 'd', '*', 's'] * 10
    colors = ['blue', 'green', 'orange', 'k', 'c'] * 10    
    cmap = plt.get_cmap('inferno_r')
    
    for ic, epsilon_chord in enumerate(epsilon_chord_ls):
    
        N_ls = np.round(eps_dr * 12.5 / epsilon_chord).astype(int)
        
        print(ic % 2, ic // 2)
        ax = axs[ic % 2, ic // 2]
        
        for im, N in enumerate(N_ls):
    
            actuator = actuatorClass(
                        N=N, 
                        chord=1,
                        L=12.5,
                        twist=6,
                        epsilon_chord=epsilon_chord,
                        U=1,
                        # ~ lookup_table='Polar2PiAlpha_AD15.dat',
                        lookup_table='NACA64_A17.dat',
                        case_name='Constant_Chord',
                        )            
                        
            label = '$\epsilon/\Delta z=$' + "{:.2f}".format(eps_dr[im])
    
            # Solve the filtered lifting line theory equations        
            actuator.solve()
    
            print('mean phi', actuator.phi.mean())
            print('mean uy',  actuator.uy.mean())
    

            # The colormap inteprolated
            c = cmap( (im+1)/ len(N_ls))
                
            # Plot the induced velocity
            ax.plot(actuator.rad / actuator.L, - actuator.uy, 
                        c=c,
                        ms=ms[im],
                        marker='o',
                        lw=2,
                        label=label)
            
        ax.grid(True)
        ax.set_title('$\epsilon/c =$' + str("{:.2f}".format(epsilon_chord)))

    # Label the bottom axes
    for ax in axs[-1, :]:
        ax.set_xlabel(r'$z/S$')
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$u_y'/U_\infty$")

    bbox = (1.03, 0.5)  # Adjust the coordinates to position the legend on the right side
    loc = 'center right'  # Set the location to the right side
    handles, labels = ax.get_legend_handles_labels()    
    fig.legend(labels=labels, loc=loc, ncol=1, bbox_to_anchor=bbox)

    plt.savefig('figure_4.pdf', bbox_inches='tight')   


def figure_6_8():
    '''
    Plot the different wings
    '''

    N = 600  # Number of points
    L = 1  # length of the wing [m]
    chord = .08 # The chord [m]
    epsilon_chord = 0.25 #[-]
    A = L * chord # Area of wing [m^2]
    twist = 6

    # Elliptic wing
    r_2 = np.linspace(-L/2, L/2, 100) 
    c_2 = np.sqrt(1-(2 * r_2/L)**2) * .08
    c_2[0]=.01
    c_2[-1]=.01
    c_2 *= A / np.trapz(c_2, r_2) # scale to match area

    # Mimic turbine blade
    r_3 = np.array([ -.5,  -.45, .5]) * L 
    c_3 = np.array([.06, .16, .05]) / 2
    c_3 *= A / np.trapz(c_3, r_3) # scale to match area
    
    cases = [    
        actuatorClass(
                    N=N, L=L,
                    chord=chord,
                    twist=twist,
                    epsilon_chord=epsilon_chord,
                    lookup_table='NACA64_A17.dat',
                    case_name='Constant Chord Wing',
                    color='tab:blue',
                    nc_file='fixed_wing/F1.nc',
                     ), 
        actuatorClass(
                    N=N, L=L,
                    twist=twist,
                    epsilon_chord=epsilon_chord,
                    lookup_table='NACA64_A17.dat',
                    case_name='Elliptic Wing',
                    chord_profile=c_2,
                    radial_profile=r_2, 
                    color='tab:orange',
                    nc_file='elliptic_wing/F1.nc',
                     ),
        actuatorClass(
                    N=N, L=L,
                    twist=twist,
                    epsilon_chord=epsilon_chord,
                    lookup_table='NACA64_A17.dat',
                    case_name='Wind Turbine Blade',
                    chord_profile=c_3,
                    radial_profile=r_3, 
                    color='tab:green',                    
                    nc_file='wind_turbine_blade/F1.nc',
                     ),
         ]

    # The figure for the induced velocities
    fig1, ax =  plt.subplots(figsize=(12, 6))

    # The figure for the wings
    fig2, axs = plt.subplots(nrows=3, ncols=1,
        sharex=True,  
        sharey=False,  
        figsize=(12, 8),
        height_ratios=[1, 1, 1],
        )
    plt.subplots_adjust( hspace=.075,)

    '''
    Induced velocity
    '''
    for case in cases:
        
        # The label for the legend
        label = case.case_name
        
        # ~ print(case.epsilon_ls / case.dr)
        print('Eps/dr min=', np.amin(case.epsilon_ls / case.dr))
        

        # Solve the filtered lifting line theory equations        
        case.solve()

        try:
            nc_file = case.nc_file
            data = ncdf.Dataset(nc_file)
            wing = data['F1']
            les_r = wing['xyz'][:, 1]
            les_uy = -wing['veff'][-1, :, 2]
            ax.plot(les_r, les_uy, 'o', ms=10, markevery=1, 
                color=case.color, label='LES Solution ' + case.case_name, 
                markeredgecolor='k',
                alpha=0.4, 
                )
        except:
            print('No LES')
            
        # Plot the induced velocity
        ax.plot(case.rad , - case.uy/case.U_ls,
                    lw=1.8,
                    label='Theory Solution ' + label,
                    path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()],
                    color=case.color)

    ax.set_xlabel(r'$r/S$')
    ax.set_ylabel(r'$u_y/U_\infty$')
    ax.legend(loc='best', )
    ax.grid()
    
    fig1.savefig('figure_8.pdf', bbox_inches='tight')

    '''
    The wings
    '''
    for i, case in enumerate(cases):
        r = [case.rad[0], *case.rad,      case.rad[-1], case.rad[0]]
        c = [0, *case.chord_ls, 0, 0]
        
        ax = axs[i]
                
        ax.plot(r, c, color='k' ) 
        ax.fill_between(r, c, label=case.case_name, color=case.color) 
        ax.set_ylabel(r'$c/S$')

        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        ax.text(.52, .11, case.case_name, ha="right", va="center", size=12,
                bbox=bbox_props)
        ax.axis('equal')

        ax.set_ylim([-.05, .18])
    
    plt.xlabel(r'$r/S$')

    fig2.savefig('figure_6.pdf', bbox_inches='tight')  
    

def figure_7(figname='figure_7.pdf'):

    
    fig = plt.figure(figsize=(12, 8))  # Adjust figsize as needed

    gs = GridSpec(1, 2, width_ratios=[1, 1])  # 1 row, 2 columns, width ratio main:inset

    ax_main = fig.add_subplot(gs[0, 0])
    ax_inset = fig.add_subplot(gs[0, 1])


    refines = [
        (-100, -100, -100, 100, 100, 100),
        (-1.28, -1.6699999999999997, -1.44, 2.44, 1.6699999999999997, 1.44),
        (-1.12, -1.5099999999999998, -1.28, 2.28, 1.5099999999999998, 1.28),
        (-0.9600000000000001, -1.3499999999999999, -1.12, 2.12, 1.349999999),
        (-0.8, -1.19, -0.9600000000000001, 1.9600000000000001, 1.19, 0.96000),
        (-0.64, -1.03, -0.8, 1.8, 1.03, 0.8),
        (-0.34, -0.65, -0.35, 1., 0.65, 0.35),
        (-0.1, -0.55, -0.22, .1, 0.55, 0.22),
        (-0.08, -0.52, -.08, .08, .52, .08),
        ]

    for n, refine in enumerate(refines):
        
        x_min = refine[0]
        x_max = refine[3]
        y_min = refine[1]
        y_max = refine[4]

        nx = 32 * 2**n
        ny = 64 * 2**n
        
        x, dx = np.linspace( -6, 10, nx + 1, retstep=True)
        y, dy = np.linspace(-16, 16, ny + 1, retstep=True)

        # Create masks for points inside the rectangle
        mask_x = (x >= x_min) & (x <= x_max)
        mask_y = (y >= y_min) & (y <= y_max)
        x = x[mask_x]
        y = y[mask_y]

        lw = .5/(n+1)
        alpha = 1 / (n+1)
        # ~ alpha = .9
        
        onesx = np.ones(len(y))
        for x1 in x:
            # ~ plt.plot(onesx * x1, y, c='k', alpha=alpha, lw=lw)
            ax_main.plot(onesx * x1, y, c='k', alpha=alpha, lw=lw)
            ax_inset.plot(onesx * x1, y, c='k', alpha=alpha, lw=lw)
                    
        onesy = np.ones(len(x))
        for y1 in y:
            # ~ plt.plot(x, onesy * y1, c='k', alpha=alpha, lw=lw)
            ax_main.plot(x, onesy * y1, c='k', alpha=alpha, lw=lw)
            ax_inset.plot(x, onesy * y1, c='k', alpha=alpha, lw=lw)
                        
        # ~ plt.plot([x.min(), x.max(), x.max(), x.min(), x.min()], 
             # ~ [y.min(), y.min(), y.max(), y.max(), y.min()], 
             # ~ 'b-', lw=2)
        ax_main.plot([x.min(), x.max(), x.max(), x.min(), x.min()],
                     [y.min(), y.min(), y.max(), y.max(), y.min()],
                     'b-', lw=2)
        ax_inset.plot([x.min(), x.max(), x.max(), x.min(), x.min()],
                      [y.min(), y.min(), y.max(), y.max(), y.min()],
                      'b-', lw=2)
                                   
            
    ax_main.set_xlabel(r'$x/D$', size=20)
    ax_main.set_ylabel(r'$z/D$', size=20)
    
    for ax in [ax_main, ax_inset]:
        ax.plot([0]*600, np.linspace(-.5, .5, 600), 'o', c='tab:red', alpha=1)
    
    ax_main.set_aspect('equal')
    ax_inset.set_aspect('equal')
    
    ax_main.set_xlim(-6, 10)
    ax_main.set_ylim(-16, 16)
    ax_inset.set_xlim(-2, 3)
    ax_inset.set_ylim(-2, 2)
    
    mark_inset(ax_main, ax_inset, loc1=2, loc2=3, fc="none", ec="0.5")
    
    plt.savefig(figname, bbox_inches='tight')

    
def figure_5():
    '''
    
    '''
    # Plot line style
    ls = '-o' 

    # Epsilon / chord
    epsilon_chord = .25
    # Wing span
    L = 12.5
    
    var = 'epsilon'
    xlabel = r'$\epsilon/\Delta z$'

    figsize=(8, 3)
    plt.figure(figsize=figsize)

    markers = ['s', 'o', '*', '+']
    # Loop through different epsilon values
    for im, epsilon_chord in enumerate([.15, .25, .5, 1]):
    
        # Number of points to ensure that we have the same eps/dr for 
        #   all cases
        # Epsilon / dr
        eps_dr = np.logspace(np.log10(.8), np.log10(15), 50) #[::-1]

        # Number of points
        N_ls = (eps_dr * L / epsilon_chord).astype(int)
        
        cases = [
            actuatorClass(
                        N=N, 
                        chord=1,
                        L=L,
                        twist=6,
                        epsilon_chord=epsilon_chord,
                        U=1,
                        # ~ lookup_table='Polar2PiAlpha_AD15.dat',
                        lookup_table='NACA64_A17.dat',
                        case_name='Constant_Chord',
                        )            
            for N in N_ls
            ]

        # Solve the equations for each case    
        for actuator in cases:

            print('Optimizing ', actuator.N)

            # Solve the filtered lifting line theory equations        
            actuator.solve()
            
            print('Optimization done')
        

        # Compute the mean of the absolute error compared to the finest case
        def error_metric(u_str, cases):
            err = []
            un = getattr(cases[-1], u_str)  # The value for the last case of the list (converged)
            # Compute the error metric for each case
            for case in cases:
                u = getattr(case, u_str)  # The variable for the current case
                
                # Mean abs error
                # ~ err.append(((u - np.interp(case.rad, cases[-1].rad, un))/u.mean()).max() * 100)
                err_1 = u - np.interp(case.rad, cases[-1].rad, un)
                abs_err = np.abs(err_1)
                norm = np.abs(un.mean())
                
                err.append( (abs_err/norm).max() * 100)
                # ~ err.append( (abs_err/norm).mean() * 100)
                
            return err
        
        phi = error_metric('phi', cases)
        uy = error_metric('uy', cases)
        G = error_metric('Gamma', cases)
        cl = error_metric('cl', cases)
        CL = [case.CL for case in cases]
        
        epsdr = [case.epsilon_ls[1]/case.dr[1] for case in cases]
        
        label = r'$\epsilon/c=$'+ str("{:.2f}".format(epsilon_chord))

        plt.plot(epsdr, CL, marker=markers[im], c='k', label=label)

    bbox = (.5, 1.2)

    plt.xscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(r"$C_L$")
    plt.axvline(x=1,)# label=r'$\epsilon/\Delta z=1$')
    plt.legend(loc='upper center', ncol=6, bbox_to_anchor=bbox)
    plt.savefig('figure_5.pdf', bbox_inches='tight')   

def table_1():
    '''
    FLLT plots for difference epsilon and resolutions
    '''
    
    # The errors as percentage
    errs = [.05, 0.01, .001]
    
    # The list of practical values of epsilon/chord
    epsilon_chord_ls = [.15, .2, .25, .3, .4, .5, 1, 2, 4]
    # ~ epsilon_chord_ls = [.15, .25, .3, 2]
    
    # Make array to store values and print table later
    eps_dr_ls = np.zeros((len(errs), len(epsilon_chord_ls)))
    max_err_ls = eps_dr_ls.copy()
    
    # ~ L = 50 #12.5
    L = 12.5
    # ~ L = 25
    
    # ~ fig, axs = plt.subplots(nrows=1, ncols=len(epsilon_chord_ls), 
    fig, axs = plt.subplots(nrows=len(epsilon_chord_ls), ncols=1, 
        sharex=True,
        sharey=True,
        figsize=(16, 8*len(errs))
        )
    
    for ic, epsilon_chord in enumerate(epsilon_chord_ls):

        # The coarsest resolution
        eps_dr = 0.5
        
        # The finest resolution        
        actuator_fine = actuatorClass(
                        N=np.round(30 * L / epsilon_chord).astype(int), 
                        chord=1,
                        L=L,
                        twist=6,
                        epsilon_chord=epsilon_chord,
                        U=1,
                        lookup_table='NACA64_A17.dat',
                        case_name='Constant_Chord',
                        )                              
        actuator_fine.solve()


        r1 = actuator_fine.rad
        # ~ u1 = actuator_fine.uy
        u1 = actuator_fine.G
        # ~ u1 = actuator_fine.cl

        ax = axs[ic]
        ax.plot(r1, u1, 'o-', c='k', ms=4)

        max_err = 100 # initialize error to large value
        
        '''
        Find the minimum error
        '''     
        for ie, err in enumerate(errs):   
            
            while max_err > err:  # Error as a percent
                
                # ~ eps_dr += 0.01
                eps_dr += 0.1
                
                # Initialize the number of points                
                N = np.round(eps_dr * L / epsilon_chord).astype(int)
                # ~ N = max(N, 3)
                
                print('New Case ' , eps_dr, N)
                
                actuator = actuatorClass(
                                N=N, 
                                chord=1,
                                L=L,
                                twist=6,
                                epsilon_chord=epsilon_chord,
                                U=1,
                                lookup_table='NACA64_A17.dat',
                                case_name='Constant_Chord',
                                )  
                                
                actuator.solve()
                
                r = actuator.rad
                # ~ u = actuator.uy
                u = actuator.G
                # ~ u = actuator.cl
                
                # Interpolate the highly resolved solution to this grid
                u_int = np.interp(r, r1, u1)
                
                # ~ max_err = np.mean(np.abs((u_int - u)/np.mean(u1)))  # mean error
                max_err = np.amax(np.abs((u_int - u)/np.mean(u1)))  # max error
                # ~ max_err = np.mean(np.abs((u_int - u)/np.mean(u1)))
                # ~ u_int = np.interp(r1, r, u)
                # ~ max_err = np.max(np.abs((u_int - u1)/np.mean(u1)))
                nw = np.argmax(np.abs((u_int - u)/np.mean(u1)))
                
                print('N ', N)
                print('Where ', nw)
                print('Where 0', np.abs((u_int - u)/np.mean(u1))[0])
                print('Where -Nw-1', np.abs((u_int - u)/np.mean(u1))[-nw-1])
                print('Where Nw', np.abs((u_int - u)/np.mean(u1))[nw])
                # ~ max_err = np.amax(np.abs((u_int - u)))
                # ~ max_err = np.mean(np.abs(u_int - u))
                
                ax.plot(r, u, '-o')
                
            ax.set_xlabel(r'$r$')
            ax.set_ylabel(r'$u_y$')
            # ~ ax.set_xlim(.95*L/2, 1.001 *L/2)

        
            eps_dr_ls[ie, ic] = eps_dr
            max_err_ls[ie, ic] = max_err

                    
    # Create LaTeX table header
    latex_table = r"""
    \begin{table}[h!]
    \centering
    \begin{tabular}{|c|""" + "|".join(["c" for _ in errs]) + r"""|}
    \hline
    $\epsilon_{chord}$ """ + "".join([f"& $\epsilon_{{Dr}}({err:.1f})$ " for err in errs]) + r""" \\
    \hline
    """
    
    # Add table rows
    for ic, epsilon_chord in enumerate(epsilon_chord_ls):
        row = f'{epsilon_chord:.2f} '
        for ie, err in enumerate(errs):
            row += f'& {eps_dr_ls[ie][ic]:.2f} '
        row += r'\\ \hline'
        latex_table += row + '\n'
    
    # Close the table
    latex_table += r"""
    \end{tabular}
    \caption{Your table caption here}
    \label{tab:example}
    \end{table}
    """
    
    # Print the LaTeX table
    print(latex_table)


    # ~ plt.show()
    plt.savefig('table1_fig.pdf')


def figure_9(figname='figure_9.pdf'):
    '''
    Compare the time to solution of the cases
    '''
    
    # The list of practical values of epsilon/chord
    epsilon_chord_ls = [.15, .25, .5, 1, 2, 4]
    
    # Make array to store values and print table later
    eps_dr_ls = np.linspace(0.75, 20, 20)
    
    L = 12.5
    
    fig, ax = plt.subplots(figsize=(12, 6))
    tmax=0
    
    for ic, epsilon_chord in enumerate(epsilon_chord_ls):

        
        # The finest resolution        
        cases = [
                    actuatorClass(
                        N=np.round(eps_dr * L / epsilon_chord).astype(int), 
                        chord=1,
                        L=L,
                        twist=6,
                        epsilon_chord=epsilon_chord,
                        U=1,
                        lookup_table='NACA64_A17.dat',
                        case_name='Constant_Chord',
                        )
                        for eps_dr in eps_dr_ls  
                        ]

        for case in cases: case.solve()
        
        N = [case.N for case in cases]
        t = [case.sol_time for case in cases]
        tmax = max(tmax, max(t))
        
        ax.plot(eps_dr_ls, t, 'o-', label='$\epsilon/c =$' + "{:.2f}".format(epsilon_chord))
            
        # Perform linear regression in log-log space
        log_eps_dr = np.log10(eps_dr_ls)
        # ~ log_eps_dr = np.log10(N)
        log_t = np.log10(t)
        slope, intercept, r_value, p_value, std_err = linregress(log_eps_dr, log_t)
        # Generate points for the fitted line
        eps_dr_fit = np.linspace(np.min(eps_dr_ls), np.max(eps_dr_ls), 100)
        t_fit = 10**(intercept + slope * np.log10(eps_dr_fit))

        # Plot the linear fit
        ax.plot(eps_dr_fit, t_fit, '--', color=ax.get_lines()[-1].get_color(),
        label=f'Fit: $t \propto (\epsilon/{{\Delta z}})^{{{slope:.2f}}}$')
            

    ax.set_xlabel(r'$\epsilon/\Delta z$')
    ax.set_ylabel('Time to Solution [s]')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='center left', bbox_to_anchor=(1, .5))
    
    fig.savefig(figname, bbox_inches='tight')


class actuatorClass():
    '''
    A class containing the properties of an actuator line
    '''
    def __init__(self, 
            N=100, 
            chord=1, 
            chord_profile=None,
            radial_profile=None,
            L=5, 
            twist=12,
            epsilon_chord=.25,
            epsilon=None,
            U=1,
            equation_uy='G',
            color='k',
            lookup_table='xf-n64012a-il-1000000.csv',
            case_name='case',
            nc_file=None,
        ):

        # Number of actuator points along the blade                        
        self.N = N
        print('N=', N)
        
        # The total length of the blade [m]
        self.L = L
        print('L=', L)
        
        # The chord value [m]
        self.chord = chord
        print('chord=', chord)

        # The chord profile [m]
        self.chord_profile = chord_profile
        print('chord_profile used')
        self.radial_profile = radial_profile
        print('radial_profile used')
        
        # The angle the blade
        self.twist = twist
        print('twist=', twist)

        # The inflow velocity
        self.U = U
        print('U=', U)
        
        # Color for the plots
        self.color=color
        
        # The equation to use to solve induced velocity
        # dG for the JFM 2019 formulation (constant epsilon)
        # G for the 2023 formulation (more general)
        self.equation_uy = equation_uy
        print('equation_uy=', equation_uy)
        
        # The value of epsilon
        self.epsilon_chord = epsilon_chord
        print('epsilon_chord=', epsilon_chord)

        # Epsilon constant
        self.epsilon = epsilon
        print('epsilon=', epsilon)

        # The filename
        self.lookup_table = lookup_table
        print('lookup_table=', lookup_table)

        # The name of the case
        self.case_name = case_name
        print('case_name=', case_name)

        # The netcdf file to access data
        self.nc_file = nc_file

        # The input file for the cl and cd tables
        self.read_lookup_table()

        # Compute the initial quantities
        self.compute_initial_quantities()

        # blank line
        print('\n')
        
    def compute_initial_quantities(self):
        '''
        Computed quantities
        '''
        # The radial positions
        # This uses a finite volume discretization were the edges are
        #   at the boundaries and all the inside points are in the 
        #   center of the control volume
        self.rad, dr = np.linspace(-self.L/2, self.L/2, self.N, retstep=True,)
        self.dr = dr * np.ones(self.N)

        # Ensure that the boundaries have the right width (dr/2)
        self.dr[0] /= 2.
        self.dr[-1] /= 2.
        # Shift the last point to make numerical integration easier
        # ~ self.rad[0] += self.dr[0]/2
        # ~ self.rad[-1] -= self.dr[-1]/2
        
        # Calculate the cell boundaries
        # ~ cell_boundaries = np.linspace(-self.L/2, self.L/2, self.N+1)
        
        # ~ # Calculate the cell centers as the average of the boundaries of each cell
        # ~ self.rad = 0.5 * (cell_boundaries[:-1] + cell_boundaries[1:])
        
        # ~ # Calculate the width of each cell (should be the same for uniform grid)
        # ~ self.dr = np.diff(cell_boundaries)
        
        # Take the midpoint of the segments
        # ~ self.rad, dr = np.linspace(-self.L/2, self.L/2, self.N + 1, retstep=True)
        # ~ self.dr = dr * np.ones(self.N)
        # ~ # Ensure that the points are the midpoint of each segment
        # ~ #  (notice that the location of the endpoints changes with N)
        # ~ self.rad = np.linspace(-self.L/2 + dr / 2, self.L/2 - dr / 2, self.N)

        # The chord as a vector
        self.chord_ls = np.ones(self.N) * self.chord
        if self.chord_profile is not None:
            self.chord_ls = np.interp(self.rad, self.radial_profile, 
                self.chord_profile, 
                # ~ left=0, right=0
                )
            # ~ self.chord_ls = gaussian_filter1d(self.chord_ls, sigma=15)
            print('Chord profile interpolated from input data')
        
        # The twist along the blade
        self.twist_ls = np.ones(self.N) * self.twist
        
        # The inflow velocity along the blade
        self.U_ls = self.U * np.ones(self.N)
        self.uy = np.zeros(self.N)

        # Compute epsilon along the blade
        if self.epsilon:
            self.epsilon_ls = self.epsilon * np.ones(self.N)
            print('Epsilon constant used throghout the blade')
        else:
            self.epsilon_ls = self.epsilon_chord * self.chord_ls # [::-1]
            # ~ self.epsilon_ls = np.maximum(self.epsilon_ls, np.ones(self.N)*.001)
            # ~ self.epsilon_ls = np.minimum(self.epsilon_ls, np.ones(self.N)*.01)
            print('Epsilon/chord used thoughout the blade')

        # ~ print('epsilon_ls=',self.epsilon_ls)

    def read_lookup_table(self):
        '''
        Read the lookup table
        '''
        
        if '.csv' in self.lookup_table:
            
            # Cl coefficient data
            data = np.loadtxt(self.lookup_table, skiprows=11, 
                delimiter=',')
                
            self.aoa_lookup = data[:,0]
            self.cl_lookup = data[:,1]

        # Read in an openfast airfoil input file            
        if '.txt' in self.lookup_table:
            # Cl coefficient data
            data = np.loadtxt(self.lookup_table, skiprows=55,)
                
            self.aoa_lookup = data[:,0]
            self.cl_lookup = data[:,1]
        
        # Old openfast input
        if '.dat' in self.lookup_table:

            # Cl coefficient data
            try:
                data = np.loadtxt(self.lookup_table, skiprows=14,)
                print('Data loaded with 14 skip')                
            except:
                data = np.loadtxt(self.lookup_table, skiprows=55,)
                print('Data loaded with 55 skip')
                
            self.aoa_lookup = data[:,0]
            self.cl_lookup = data[:,1]
        
        # Inteprolation function
        self.interp_aoa = interp1d(self.aoa_lookup, self.cl_lookup)
        
    def solve(self):
        '''
        Solve the equation to obtain the quantities along the blades
        '''

        # ~ nc_file = 'wind_turbine_blade/F1.nc'
        # ~ print(nc_file)
        # ~ data = ncdf.Dataset(nc_file)
        # ~ print(data)
        # ~ wing = data['F1']
        # ~ print(wing)
        # ~ les_r = wing['xyz'][:, 1]
        # ~ les_uy = wing['aoa'][-1, :]
        # ~ les_uy = -wing['veff'][-1, :, 2]
        # ~ les_uy = wing['body_force'][-1, :, 2]
        # ~ les_uy = 1./2. * wing['chord'][:] * wing['cl'][-1,:] * (wing['veff'][-1, :, 0]**2 + wing['veff'][-1, :, 2]**2 ) 


        def phi_function(phi):
            '''
            This is a function used to find the root
            '''
            # sine and cosine
            sinphi = np.sin(np.deg2rad(phi))
            cosphi = np.cos(np.deg2rad(phi))
            tanphi = np.tan(np.deg2rad(phi))

            # ~ phi = np.mod(phi + 180, 360) - 180
            # ~ phi = np.clip(phi, -90, 90)
            phi = np.clip(phi, -150, 150)
            # ~ phi = np.clip(phi, -180, 180)
            # ~ phi = np.clip(phi, -70, 70)
            # ~ phi = np.clip(phi, -90, 0)

            # Compute the total angle of attack
            aoa_t = phi + self.twist_ls
            aoa_t = np.mod(aoa_t + 180, 360) - 180

            # Determine the lift coefficient
            # ~ cl = np.interp(aoa_t,  self.aoa_lookup, self.cl_lookup)
            cl = self.interp_aoa(aoa_t)
            
            # Test Matt's flaps idea
            # ~ cond = (self.rad>= -.25*self.L) & (self.rad<= .25*self.L)
            # ~ cl[cond ] += .2

            # The relative velocity
            # ~ W = np.sqrt(self.U_ls**2 + self.uy**2) 
            # ~ W = self.U * np.sqrt(1 + tanphi**2) 
            # ~ W = self.U_ls / np.abs(cosphi) 
            W = self.U_ls / cosphi 
            # ~ W = self.U_ls

            # Compute the circulation times U
            G = 1/2 * cl * self.chord_ls * W**2 # * cosphi
            # ~ G[0]/=2
            # ~ G[-1]/=2

            # ~ G = np.interp(self.rad, les_r, les_uy)
            # ~ dG = np.diff(G, prepend=G[0]) / self.dr

            # Compute the gradient of the circulation
            dG = np.gradient(G) / np.gradient(self.rad)
            # ~ dG = gaussian_filter1d(np.gradient(G, self.rad, ), sigma=5)
            # ~ dG = np.gradient(G, self.rad,) # edge_order=2)
            # ~ dG = np.gradient(G, self.rad,) # edge_order=2)
            # ~ fd = G / self.chord_ls
            # ~ dG = np.gradient(fd, self.rad, )* self.chord_ls + np.gradient(self.chord_ls, self.rad) * fd  
            # ~ dG[0] = G[0] / self.dr[0] #* 2
            # ~ dG[-1] = - G[-1] / self.dr[-1] #* 2
            # ~ dG = np.zeros(self.N)
            # ~ for i in range(1, self.N-1):
                # ~ dG[i] = (G[i+1] - G[i-1]) / (self.rad[i+1] - self.rad[i-1])
            dG[0] = G[0] / self.dr[0]
            dG[-1] = -G[-1] / self.dr[-1]
            # ~ dG[0] *= 2
            # ~ dG[-1] *=2 
            
            # Fifth order fd
            # ~ for i in range(2, self.N-2):
                # ~ dG[i] = (-G[i+2] +8 * G[i+1] -8*G[i-1] + G[i-2]) / (self.dr[i] * 12)
                 
            # Compute more exact formulation for FLLT
            def f_integral(y):
                epsilon = self.epsilon_ls
                exp = np.exp(-y**2 / epsilon**2)
                
                g = -G / epsilon**2 * (
                    # term f1
                    (exp)  
                    +
                    # term f2
                    (epsilon**2 * (exp - 1) / (2* y**2 + 1e-36))
                    ) 
                    
                cond = (y==0.)
                # ~ cond = np.isclose(0,y)
                g[cond] = - .5 * G[cond] / (epsilon[cond]**2)
                # ~ g[cond] = 0
                
                return g
                
            def f_0():
                '''
                Comptue the singularity of the integral based on 
                    formulations from Meneveau
                '''
                # ~ i = np.argwhere(y==0.)
                # ~ if (i>0) and (i<len(y)-1):
                epsilon = self.epsilon_ls
                g = np.zeros(self.N)
                # Loop through inner points
                for i in range(1, self.N-1):
                    y = self.rad - self.rad[i]
                    t1 = 1/(2 * y[i-1]/2) * (1-np.exp(-(y[i-1]/2)**2/epsilon[i]**2))
                    t2 = 1/(2 * y[i+1]/2) * (1-np.exp(-(y[i+1]/2)**2/epsilon[i]**2))
                    g[i] = -1/(2*np.pi) * G[i] * (t2-t1)
                    
                y = self.rad - self.rad[0]
                g[0] = -1/(2*np.pi) * G[0] * 1/(2 * y[1]/2) * (1-np.exp(-(y[1]/2)**2/epsilon[0]**2))

                y = self.rad - self.rad[-1]
                g[-1] = 1/(2*np.pi) * G[-1] * 1/(2 * y[-2]/2) * (1-np.exp(-(y[-2]/2)**2/epsilon[-1]**2))

                # ~ plt.figure(7)
                # ~ plt.plot(g, 'o-')                    
                # ~ plt.show()
                
                return g

            if self.equation_uy == 'dG':
                # Compute the induced velocity
                uy = np.array([ -np.sum( dG * uy_function( self.rad[i] - self.rad, 
                        self.epsilon_ls) * self.dr / W)  for i in range(self.N)])

            elif self.equation_uy == 'G':
                # Compute induced velocity following integration
                uy =  1/(2 * np.pi) * np.array([ 
                        integrate.trapezoid(f_integral(self.rad-self.rad[i]), x=self.rad) 
                        # ~ integrate.simpson(f_integral(self.rad-self.rad[i]), x=self.rad, even='simpson') # it is important to use the even option 
                        # ~ np.sum(f_integral(self.rad-self.rad[i])*self.dr)
                        # ~ integrate.romb(f_integral(self.rad-self.rad[i]), dx=self.rad[1]-self.rad[0])
                        # ~ integrate.fixed_quad(lambda x: f_integral(x - self.rad[i]), np.min(self.rad), np.max(self.rad), n=len(self.rad))[0]
                        for i in range(self.N)]) 
                        
            # Add the contribution from the rest of the integral
            # ~ uy += f_0()
                            
            self.W = W
            self.uy = uy
            self.dG = dG
            self.G = G
            self.cl = cl
            self.aoa = aoa_t
            self.CL = np.sum(self.G * self.dr) / (1/2 * self.U**2 * np.sum(self.dr * self.chord_ls)) 
    
            # The function to optimize (should be zero)
            # This is a similar application of Ning's 2014 Wind Energy BEM paper
            f = self.U_ls * sinphi - uy * cosphi
            # ~ f = sinphi / uy - cosphi / self.U_ls
            # ~ f = -self.U_ls * sinphi + uy * cosphi
            # ~ f = tanphi - uy / self.U_ls
            # ~ f = 

            return f
 
        def phi_min(phi):
            '''
            Function to call within a minimizer
            returns single float
            '''
            f = phi_function(phi)
            return np.sum(f**2)
 
        '''
        Solve the equation using Dries' method
        '''
        # Initial condition for phi
        # ~ phi_0 = self.twist_ls
        # ~ phi_0 = np.linspace(-45, 25, self.N)
        # ~ phi_0 = np.ones(self.N) * 40
        phi_0 = np.zeros(self.N)  # Dries suggestion - flow angle of zero means alpha = beta

        # ~ nc_file = 'wind_turbine_blade/F1.nc'
        # ~ data = ncdf.Dataset(nc_file)
        # ~ wing = data['F1']
        # ~ les_r = wing['xyz'][:, 1]
        # ~ les_uy = wing['veff'][-1, :, 2]
        # ~ p = np.arctan2(wing['veff'][-1, :, 2], wing['veff'][-1, :, 0])
        # ~ p = np.rad2deg(p)
        # ~ phi_0 = np.interp(self.rad, les_r, p)
        # ~ plt.plot(phi_0)
        # ~ plt.show()


        # Solve the equation for the flow angle
        # ~ self.phi = scipy.optimize.anderson(phi_function, phi_0)
        # ~ self.phi = scipy.optimize.anderson(phi_function, phi_0, f_rtol=1e-15, f_tol=1e-15,)
        # ~ self.phi = scipy.optimize.anderson(phi_function, phi_0, f_tol=1e-12)
        
        # Other solve methods
        
        # Takes a long time to converge
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='lm').x 
        # Takes longer to converge (not too bad)
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='hybr').x 
        # Pretty good
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='broyden1').x 
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='broyden2').x 
        
        # This is the fastest algorithm
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='anderson').x 
        # Found incorrect solution for all resolutions (do not use)
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='linearmixing').x 
        # Found incorrect solution for all resolutions (do not use)
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='diagbroyden').x 
        # Very slow AND wrong solutions(do not use)
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='excitingmixing').x 

        # ~ options = {'ftol':1e-25, 'M':100, 'line_search': 'cheng'}
        # ~ options={}
        
        # Super fast! (faster than anderson)
        start_time = time.time()
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='krylov', tol=1e-12).x 
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='df-sane', options=options).x 
        self.phi = scipy.optimize.root(phi_function, phi_0, method='df-sane',).x 
        # ~ self.phi = scipy.optimize.root(phi_function, phi_0, method='df-sane', options={'fatol': 1e-16}).x 
        
        self.sol_time = time.time() - start_time
        print("--- %s seconds to solve fll equations---" % (self.sol_time))     
        print()   


        # Evaluate the solution
        phi_function(self.phi)
            
        # Gamma
        self.Gamma = 1/2 * self.cl * self.chord_ls * np.sqrt(self.U_ls**2 + self.uy**2)
    
# The function inside the integral
def uy_function(z, epsilon):
    '''
    Function to compute induced velocity
    '''
    # ~ eps = .0039/2  * np.sign(-z) + 1e-16
    eps = 1e-16
    f = 1 / (4 * np.pi * z + eps ) * (1 - np.exp(-(z)**2 / epsilon**2))
    # ~ f[(z==0]= 0
    # ~ f[(z**2 / epsilon**2) > 16] *= 0
    return f
    # ~ return 1 / (4 * np.pi * z + 1e-20 ) * (1 - np.exp(-z**2 / epsilon**2))
    # ~ return np.divide(1, (4 * np.pi * z ), where=z!=0, out=np.zeros_like(z)) * (1 - np.exp(-z**2 / epsilon**2))


if __name__ == '__main__':
	main()
