##Code to read interpolated simulated long light curve (already in sec) and create power spectrum

import os
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
# import tkinter
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
# from scipy.signal import savgol_filter as sgf
# from astropy.timeseries import LombScargle

#########################################################################################################################
#########################################################################################################################

## This section contains the equations used for fits to data

def linear(x,m,b):
    return m*x+b 

def piecewise_linear(x,x_turn,m1,m2,b1,b2):
    return np.piecewise(x,[x<x_turn,x>x_turn],[lambda x:m1*x+b1, lambda x:m2*x+b2])

def lorentz(x,cen,gamma):
#   return gamma/(np.pi*((x-cen)**2 + gamma**2))
    return (0.5*gamma/np.pi)/((x-cen)**2 + (0.5*gamma)**2)

def combolorentz(x,cen,gamma,m,b):
    return linear(x,m,b)+lorentz(x,cen,gamma)

def linearknee(x,x_knee,m,b):
    return b+m/2*np.log10(1+(x/x_knee)**2)


# def power_law(x, a, b):
#   return a*np.power(x, b)

# def chisquare_test():
#   return()

####################################################################################################################################
####################################################################################################################################

## This section contains functions for finding points within the PS data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_intersection(m1,m2,b1,b2):
  
    xi = (b2-b1) / (m1-m2)
    yi = m1 * xi + b1

    return xi
####################################################################################################################################
####################################################################################################################################


def correct_data(df, datapts, subdir, target, num_lcs, bkg, redshift):
  
    ## Sequence of finding gaps in data, linearly interpolating between gaps, and plotting of new figures.

    ## Set intial values from the light curve dataframe 
    ## Plotting values first then write to .dat

    single_lc_length = int((len(df[0])/num_lcs)+0.5)
    
    startcad = single_lc_length
    endcad = startcad+single_lc_length
    
    ## Initialize arrays for plotting using REAL lightcurve
    
    linear_fit = np.polyfit(df[0][0:startcad],df[1][0:startcad],1)
    flattened_reallc = np.subtract(df[1][0:startcad],linear_fit[0]*df[0][0:startcad])
    
    ## Find mean subtracted flux 
    
    # mean_arr = []
    mean_flux = np.mean(df[1][0:startcad])
    meansubflux = np.subtract(flattened_reallc,np.mean(flattened_reallc))
    
    
    full_time = list(np.divide(df[0][0:startcad],(1+redshift)))
    
    full_flux = list(meansubflux)
    full_err = list(df[2][0:startcad])
    
    dft = np.fft.fft(meansubflux)  #calculate power spectrum...
    ps = np.abs(dft)**2
    
    
    sampspace = full_time[1]-full_time[0]
    fspace = np.fft.fftfreq(len(full_time), d=sampspace)
    # fspace = np.fft.fftfreq(lc_time_final.size, d=sampspace)
    
    
    real_ps = np.column_stack((fspace,ps))
    pos_freq_lcreal = [i for i,j in real_ps if i>0]
    # print(len(pos_freq_lcsim))
    pos_pow_lcreal = [j for i,j in real_ps if i>0]
    
    A_rms2 = (2*sampspace)/(mean_flux**2 * len(pos_pow_lcreal)) ##VdK normalization
    pos_pow_lcreal = np.multiply(A_rms2,pos_pow_lcreal)
    
    poisson_vdk_arr = [2*(mean_flux+bkg)/(mean_flux**2)]
    
    
    full_ps = np.column_stack((pos_freq_lcreal,pos_pow_lcreal))

    
    # badmask = np.where(np.isfinite(full_err) == False)[0]
    badmask = np.where(np.array(full_err) == 0)[0]
    
    
    for k in range(1,num_lcs):

        # print("k: "+str(k))
        # print("startcad = %i"%startcad)
        # print("endcad = %i"%endcad)
        
        
        t_slice = np.array(df[0][startcad:endcad])
        # print("len(t_slice): %i"%len(t_slice))
        lc_slice = np.array(df[1][startcad:endcad])
        mean_flux = np.mean(lc_slice)
        err_slice = np.array(df[2][startcad:endcad])
        # print("len(lc_slice): %i"%len(lc_slice))
        # print()
        # timeax = np.array(range(len(lc_slice)))
        
        # badmask_setzero = badmask_unique - badmask_unique[0]
        
        for i in badmask:
            # print(i)
            lc_slice[i] = 0
            err_slice[i] = 0
        
        nozero_full = np.column_stack((t_slice,lc_slice)) #timeax
        
        nozero_time = [i for i,j in list(nozero_full) if j != 0]
        nozero_flux = [j for i,j in list(nozero_full) if j != 0]
        
        good_lc_only = np.column_stack((nozero_time,nozero_flux))
        
        patch = np.zeros(2).reshape((1,2))
        
        timearray = np.array(nozero_time)
        fluxarray = np.array(nozero_flux)
        
        for i in range(0, len(badmask)):                           #Loop over each bad cadence...
            patchflux = np.interp(badmask[i],timearray,fluxarray)  #Calculate interpolated value at the bad cadence value.
            newvals = [badmask[i],patchflux]
            patchrow = np.column_stack(newvals)
        
            patch = np.concatenate([patch,patchrow])                       #Tack this new interpolated row onto the patch.
        
        patch = np.delete(patch, (0), axis=0)
        
        combine_patch = np.concatenate([good_lc_only,patch])               #Join the good array and the patch.
        combine_patch_sort = combine_patch[combine_patch[:,0].argsort()]
        # print(combine_patch_sort)
        
        patched_lc_time = combine_patch_sort[:,0]
        patched_lc_flux = combine_patch_sort[:,1]
        
        linear_fit = np.polyfit(patched_lc_time,patched_lc_flux,1)
        flattened_simlc = np.subtract(patched_lc_flux,linear_fit[0]*patched_lc_time)
        
        lc_time_final = patched_lc_time
        lc_flux_final = flattened_simlc
        # print("len(lc_flux_final): %i"%len(lc_flux_final))
        
         ## Find mean subtracted flux and redshift correct time
        
        zcorr_time = np.divide(lc_time_final,(1+redshift))
        # mean_arr.append(np.mean(lc_flux_final))
        meansubflux = np.subtract(lc_flux_final,np.mean(lc_flux_final))
        
        # print("len(meansubflux): %i"%len(meansubflux))
        
        full_time = np.concatenate((full_time,zcorr_time))
        full_flux = np.concatenate((full_flux,meansubflux))
        full_err = np.concatenate((full_err,err_slice))
        
        
        ##############################################################################

        ## PSD analysis
    
        
        dft = np.fft.fft(meansubflux)  #calculate power spectrum...
        ps = list(np.abs(dft)**2)
        
        fspace = np.fft.fftfreq(zcorr_time.size, d=sampspace)
        # fspace = np.fft.fftfreq(lc_time_final.size, d=sampspace)
        
        
        sim_ps = np.column_stack((fspace,ps))
        # pos_freq_lcsim = [i for i,j in full_ps if i>0]
        # print(len(pos_freq_lcsim))
        pos_pow_lcsim = [j for i,j in sim_ps if i>0]
        A_rms2 = (2*sampspace)/(mean_flux**2 * len(pos_pow_lcsim)) ##VdK normalization
        pos_pow_lcsim = np.multiply(A_rms2,pos_pow_lcsim)
        
        full_ps = np.column_stack((full_ps,pos_pow_lcsim))
        
        
        poisson_vdk_arr.append(2*(mean_flux+bkg)/(mean_flux**2))
        
            
        startcad = startcad+single_lc_length
        endcad = startcad+single_lc_length
        
    
    poisson_vdk = np.mean(poisson_vdk_arr)
    
    np.savetxt(subdir+'/'+target+'_sim_500psarray.dat',full_ps,delimiter=' ')
    
    full_ps = pd.DataFrame(full_ps)
    print(full_ps)
    
    # print("len(full_flux): %i"%len(full_flux))
    corrected_data = pd.DataFrame(np.column_stack((full_time, full_flux, full_err)))
    corrected_data.replace(np.nan, 0.0, inplace=True)
    
    ##Create a .dat file for the newly interpolated data without overwriting the orgininal data
    interp_data = open(os.path.join(subdir,target+'_full_sim_Corrected_lc.dat') , "w")
    
    for i in range(0,len(corrected_data[0])):
        # print(i)
        # print("%f %.10f %.10f\n" %(corrected_data[0][i],corrected_data[1][i],corrected_data[2][i]))
        interp_data.write("%.10f %.10f %.10f\n" %(corrected_data[0][i],corrected_data[1][i],corrected_data[2][i]))
        
    ## Close new .dat file then print and save new interpolated light curve
    interp_data.close()
    
    
    return(corrected_data,full_ps,poisson_vdk)
    
####################################################################################################################################
####################################################################################################################################

def powerspec(subdir,target,full_data, bkg, mean_arr):
  
    full_time = full_data[0]
    full_flux = full_data[1]
    full_err = full_data[2]
    
    mean_flux = np.mean(mean_arr)
    
    sampspace = (full_time[1]-full_time[0])
    #   print('Sampling spacing is %f seconds' %sampspace)
    
    freq = np.fft.fftfreq(len(full_time), d = sampspace)
    

    dft = np.fft.fft(full_flux)
    ps = np.abs(dft)**2

    A_rms2 = (2*sampspace)/(mean_flux**2 * len(full_flux)) ##VdK normalization
    # A_rms2 = 1
    # print('RMS squared normalization for mean sub flux PS is %.10f' %A_rms2)
    normal_ps = A_rms2*ps

    poisson_vdk = 2*(mean_flux+bkg)/(mean_flux**2)
    

    stacked_data = pd.DataFrame(np.column_stack((freq, normal_ps)))
    # print(freq)
    
    ## Remove all negative freq s.t. f > 0 only
    negfreq = np.where(freq <= 0)
    reduced_ps = stacked_data.drop(negfreq[0])
    # print(reduced_ps)


    return(reduced_ps, poisson_vdk)
    
####################################################################################################################################
####################################################################################################################################

def findnoise(reduced_ps, poisson):

    ## f at which noise dominates is estimated by creating a broken powerlaw fit where the high-f portion flattens to a slope of zero and returning the turn frequency
    freq = np.log10(reduced_ps[0])
    power = np.log10(reduced_ps[1])
    
    for i in range(1,500):
        freq = np.concatenate((freq,np.log10(reduced_ps[0])))
        power = np.concatenate((power,np.log10(reduced_ps[i+1])))

    # print(len(power))

    noisefreq = 1.0

    meansub_pars, _ = curve_fit(piecewise_linear,xdata=freq,ydata=power, bounds=((-4.8,-np.inf,-0.0001,-np.inf,-np.inf),(-3.6,np.inf,0,np.inf,np.inf)), maxfev=99999) #p0=(-4,-2,0,power[0],np.log10(np.nanmean(power[-50:]))),
    noisefreq = 10**meansub_pars[0]
    
 
    return(noisefreq)
####################################################################################################################################
####################################################################################################################################


def rebinning(subdir,target,reduced_ps, noisefreq):
    
    ## Rebinning:

    numbin = 26 #set number of total frequency bins to have.
    
    freq = list(reduced_ps[0])
    # print(freq)
    # power = list(reduced_ps[1])
    
    minfreq = np.min(freq)
    maxfreq = np.max(freq)
    f_bins = np.logspace(np.log10(minfreq), np.log10(maxfreq), num=numbin) #compute bins in frequency space
    lenbins = len(f_bins)
    # print(f_bins)

    binpow = np.zeros(lenbins-1)  #initialize binned power array

    binfreq = np.zeros(lenbins-1) #initialize binned frequency array
    binstd = np.zeros(lenbins-1) #initialize binned stdev array

    binstart_arr = np.zeros(lenbins)
    binend_arr = np.zeros(lenbins)
  

    for n in range(1,lenbins):
        binstart = freq.index(find_nearest(freq,f_bins[n-1]))
        binend = freq.index(find_nearest(freq,f_bins[n]))

        binstart_arr[n]=binstart
        binend_arr[n]=binend
        
        dif = int(binend - binstart)

        if np.abs(dif) == 1 or np.abs(dif) == 0:
            
            binfreq[n-1] = freq[n-1]
            # print(binfreq[n-1])

            if binstart_arr[n] <= binstart_arr[n-1] and n > 1:
                binstart = int(binstart_arr[n-1]+1)
                if binfreq[n-1] <= binfreq[n-2]:
                    binfreq[n-1] = freq[n]
                
            if binstart_arr[n] < binend_arr[n-1] and n > 1:
                binstart = int(binend_arr[n-1])
                if binfreq[n-1] <= binfreq[n-2]:
                    binfreq[n-1] = freq[n]
                
            binend = int(binstart + dif)
            binstart_arr[n] = binstart
            binend_arr[n] = binend
            
            power = []
            for i in range(0,500):
                power.append(reduced_ps[i+1][n-1])
        
            binpow[n-1] = np.mean(power)
            binstd[n-1] = np.std(power)

    
        else:

            if binstart_arr[n] <= binstart_arr[n-1]:
                binstart = int(binstart_arr[n-1]+1)
                
            if binstart_arr[n] < binend_arr[n-1]:
                binstart = int(binend_arr[n-1])

            binend = int(binstart + dif)    
            
            f_range = freq[binstart-1:binend+1]
            f_avg = np.mean(f_range)
            binfreq[n-1] = f_avg
            
            power = []
            for i in range(0,500):
                power.append(reduced_ps[i+1][binstart-1:binend+1])
        
            # a_range = power[binstart-1:binend+1]

            binpow[n-1] = np.mean(power)
            binstd[n-1] = np.std(power)
    
    # print("len(binpow) = %i"%len(binpow))
    # print(binstderr)
    # print(binstart_arr)
        
    # print(binend_arr)

    # logbinfreq = [log10(i) for i in binfreq]
    # logbinpow = [log10(i) for i in binpow]
  
    binnedps = pd.DataFrame(np.column_stack((binfreq,binpow)))
    np.savetxt(subdir+'/'+target+'_sim_logbinnedps.dat',np.column_stack((np.log10(binfreq),np.log10(binpow),np.log10(binstd))),delimiter=' ')

    ## Manual option to reset noise frequency based on rebinned data
    noise_good = False

    while noise_good == False:
        
        plt.figure(figsize=(12,6))
        # for i in range(1,500):
        #     plt.scatter(np.log10(reduced_ps[0]), np.log10(reduced_ps[i]), color='b')
        plt.scatter(np.log10(reduced_ps[0]), np.log10(reduced_ps[1]), color='b')
        plt.scatter(np.log10(reduced_ps[0]), np.log10(reduced_ps[250]), color='b')
        plt.scatter(np.log10(reduced_ps[0]), np.log10(reduced_ps[500]), color='b')
        plt.errorbar(np.log10(binfreq), np.log10(binpow), np.log10(binstd), color='k', linestyle='solid', label = "binned data")
        # plt.scatter(np.log10(binfreq), np.log10(binpow), s=55, color='k', marker= '|')
        plt.axvline(np.log10(noisefreq), color= 'b', linestyle = 'dotted', label = chr(957)+'$_{noise}$ ~ 10$^{%.2f}$ Hz'%np.log10(noisefreq))
        plt.legend()
        plt.show()

        # new = input("Would you like to adjust window length or polyorder? [y/n]")
        new = "n"
        new = input("Would you like to adjust noise frequency for the mean-subtracted data? [y/n]")

        if new == "y" or new=="Y" or new =="yes" or new=="YES":
            print("Enter the new noise frequency in logarithmic form (e.g. for 10^-5, just enter -5)")
            noisefreq = 10**float(input("Noise freq. for meansub data (previously 10^%f): "%np.log10(noisefreq)))
        else:
            noise_good = True
            break

    # print(binnedps[0])

    # logbinnedps = np.column_stack((logbinfreq,logbinpow))
    # np.savetxt('logbinnedps.dat',logbinnedps,delimiter=' ')
    # print(binnedps)
    return(binnedps, binstd, noisefreq)
#####################################################################################################################################################
#####################################################################################################################################################

def find_stdev(error_df):
    
    # print(error_df)
    stdev = []
    
    for i in range(0,25):
        stdev.append(np.std(error_df[i]))
        
    # print(stdev)
    
    return stdev
    
#####################################################################################################################################################
#####################################################################################################################################################

def powerlaw_fits(binnedps, noisefreq, binstd):
    
    
    binfreq = np.array(np.log10(binnedps[0]),dtype=float)
    binpow = np.array(np.log10(binnedps[1]),dtype=float)
    
    noisefreq = np.log10(noisefreq)
    ## Get slices of binned data based on determined noise threshold

    fitfreq_max = list(binfreq).index(find_nearest(binfreq, noisefreq))
    # print(fitfreq_max)
    
    fitfreq = binfreq[0:fitfreq_max]
    fitpow = binpow[0:fitfreq_max]
    fitsigma = binstd[0:fitfreq_max]
    
    ##########################################################################################################################################
    
    ## Perform single power-law fit and chi-square goodness of fit test, check p-value for p>0.25 for 25% significance level

    plaw_pars, _ = curve_fit(f=linear, xdata=fitfreq, ydata=fitpow, sigma = fitsigma, absolute_sigma = True, bounds=((-4,-np.inf),(-0.5,np.inf)), maxfev=2000)
    # print(pars[1])
    plaw_chi2, plaw_pvalue = stats.chisquare(linear(fitfreq,*plaw_pars),f_exp=fitpow, ddof=len(plaw_pars), axis = 0)

    plaw_res = fitpow - linear(fitfreq, *plaw_pars)
    # print(res)

    #######################################################################################################################################
   
    ## Perform broken power-law fit and chi-square goodness of fit test, check p-value for p>0.25 for 25% significance level
  
    ## Intitial guesses 
    brkn_pw0= [-5.5,-1,-2,1,-1]

    fslice = []
    pslice = []
    
    for i in range(0,len(fitfreq)):
        if fitfreq[i] > brkn_pw0[0] and fitfreq[i] < noisefreq:
            fslice.append(fitfreq[i])
            pslice.append(fitpow[i])
    
    fslice = np.array(fslice)
    pslice = np.array(pslice)
    slice_params = np.polyfit(fslice,pslice,1)
    brkn_pw0[0] = fslice[0]
    brkn_pw0[2] = slice_params[0]
    brkn_pw0[4] = slice_params[1]

    try:
        brkn_pars, _ = curve_fit(piecewise_linear,xdata=fitfreq,ydata=fitpow, p0=brkn_pw0, sigma = fitsigma, absolute_sigma = True, bounds=((-6.2,-1,-3,0,-np.inf),(-5,0,-1,np.nanmax(fitpow)+1,np.inf)), maxfev=999999)
    # except RuntimeError:
    #     pars, _ = curve_fit(piecewise_linear,xdata=fitfreq,ydata=fitpow, p0=pw0, bounds=((-6,-0.5,-3,0,-np.inf),(-5,0,-0.5,np.nanmax(fitpow)+1,np.inf)), maxfev=999999)
    except ValueError:
        brkn_pars, _ = curve_fit(piecewise_linear,xdata=fitfreq,ydata=fitpow, p0=brkn_pw0, sigma = fitsigma, absolute_sigma = True, bounds=((-6.2,-np.inf,-np.inf,-np.inf,-np.inf),(-5,np.inf,np.inf,np.inf,np.inf)), maxfev=5000)
    
    brkn_chi2, brkn_pvalue = stats.chisquare(piecewise_linear(fitfreq,*brkn_pars),f_exp=fitpow, ddof=len(brkn_pars), axis = 0)
    # chi2, pvalue = stats.chisquare(linear(fitfreq[hfreq_min:],pars[2],pars[4]),f_exp=fitpow[hfreq_min:]-1, ddof=len(pars),axis = 0)
    brkn_res = fitpow - piecewise_linear(fitfreq, *brkn_pars)
    
  
    ####################################################################################################################################################
    
    ## Perform combo-Lorentzian fit and chi-square goodness of fit test, check p-value for p>0.25 for 25% significance level

    ## Intitial guesses 
    meansub_cen = find_intersection(*brkn_pars[1:])

    lrntz_pw0= [meansub_cen,0.3,plaw_pars[0],plaw_pars[1]]


    try:
        lrntz_pars, _ = curve_fit(combolorentz, xdata=fitfreq, ydata=fitpow, p0=lrntz_pw0, sigma = fitsigma, absolute_sigma = True, bounds=((-6.2,-np.inf,-3,0),(-5.5,np.inf,0,np.nanmax(fitpow)+2)), maxfev=999999)
    # except RuntimeError:
    #     pars, _ = curve_fit(combolorentz, xdata=fitfreq, ydata=fitpow, p0=pw0, bounds=((-6,-0.5,-3,0,-np.inf),(-5,0,-0.5,np.nanmax(fitpow)+1,np.inf)), maxfev=999999)
    except ValueError:
        lrntz_pars, _ = curve_fit(combolorentz, xdata=fitfreq, ydata=fitpow, sigma = fitsigma, absolute_sigma = True, bounds=((-6.2,-np.inf,-np.inf,-np.inf),(-5.3,np.inf,np.inf,np.inf)), maxfev=5000)
    
    lrntz_chi2, lrntz_pvalue = stats.chisquare(combolorentz(fitfreq,*lrntz_pars),f_exp=fitpow, ddof=len(lrntz_pars),axis = 0)
    lrntz_res = fitpow - combolorentz(fitfreq, *lrntz_pars)
    
    
    ##################################################################################################################################################
    ## Perform knee-bend fit and chi-square goodness of fit test, check p-value for p>0.25 for 25% significance level

    ## frequency must not be lograithmic, logarithmic operation occurs in fit;
    ## knee-bend method adapted from  Uttley et al. 2002

    bendfreq = np.power(10,fitfreq)
    ## Intitial guesses 
    
    bend_pw0= [10**brkn_pars[0], -1*brkn_pars[2], brkn_pars[3]]


    try:
        bend_pars, _ = curve_fit(linearknee, xdata=bendfreq, ydata=fitpow, p0=bend_pw0, sigma = fitsigma, absolute_sigma = True, bounds=((10**np.nanmin(fitfreq),-3,0),(10**-5.5,-1,np.nanmax(fitpow)+2)), maxfev=999999)
    # except RuntimeError:
    #     pars, _ = curve_fit(linearknee, xdata=fitfreq, ydata=fitpow, p0=pw0, bounds=((-6,-0.5,-3,0,-np.inf),(-5,0,-0.5,np.nanmax(fitpow)+1,np.inf)), maxfev=999999)
    except ValueError:
        bend_pars, _ = curve_fit(linearknee, xdata=bendfreq, ydata=fitpow, sigma = fitsigma, absolute_sigma = True, bounds=((10**np.nanmin(fitfreq),-np.inf,-np.inf),(10**-4.8,0,np.inf)), maxfev=5000)
    
    bend_chi2, bend_pvalue = stats.chisquare(linearknee(bendfreq,*bend_pars),f_exp=fitpow, ddof=len(bend_pars),axis = 0)
    bend_res = fitpow - linearknee(bendfreq, *bend_pars)
    

    ##################################################################################################################################################

    chi2_all = pd.DataFrame(np.column_stack((plaw_chi2, brkn_chi2, lrntz_chi2, bend_chi2)))
    pvalue_all = pd.DataFrame(np.column_stack((plaw_pvalue, brkn_pvalue, lrntz_pvalue, bend_pvalue)))
   
    return(fitfreq, plaw_pars, brkn_pars, lrntz_pars, bend_pars, plaw_res, brkn_res, lrntz_res, bend_res, chi2_all, pvalue_all)
#####################################################################################################################################################
#####################################################################################################################################################


#####################################################################################################################################################
#####################################################################################################################################################

## Begin by setting the directory in which the target files are kept as rootdir

# rootdir = '/mnt/c/Users/ryned/Dropbox/My PC (DESKTOP-14FS8VG)/Desktop/AGNstudy/LightCurves/ProbablyGood/'
# rootdir = '/mnt/c/Users/ryned/Dropbox/My PC (DESKTOP-14FS8VG)/Desktop/AGNstudy/LightCurves/Interesting/'
# rootdir = '/mnt/c/Users/ryned/Dropbox/My PC (DESKTOP-14FS8VG)/Desktop/AGNstudy/LightCurves/Recovered_disagreement/'
rootdir = '/users/rdingler/AGNstudy/Sim_Results/'

num_lcs = 500

## Counting as preventative measure when only wanting to do a limited number of files. Disable if all files should be read.
count = 0

## Loop through regression_program_output/ directory and target files to find light curve .dat files
for subdir, dirs , files in os.walk(rootdir):
    
    count += 1
     
    ## Get target name from subdirectory name
    target = os.path.basename(subdir)
    
    if target != "" and target != "eleanor_apertures" and target != "quaver_lc_apertures" and target != "flux_distributions" :
        print(target)
        bkg = int(input('Appox of average background of '+target+' from FITS: '))
        z = float(input('Appox of redshift of '+target+' from NED: '))
        
    # pipeline_lc = np.loadtxt('/users/rdingler/AGNstudy/LightCurves/Analysis/'+target+'/'+target+'_full_regressed_interpolated_lc.dat')


    # time_lc = np.array([x for x,y,z in pipeline_lc])  
    # flux_lc = np.array([y for x,y,z in pipeline_lc])
    # error_lc = np.array([z for x,y,z in pipeline_lc])
    
    ## Initialize arrays for plotting, must do before going through files in order to compare hybrid and pca5 data
    full_data = []
    reduced_ps = []  
    binnedps  =[]

    # LSfreq = []
    # LSpow = []

    poisson = 1.0
    # poisson_vdk_LS = 1.0

    noisefreq = 1.0
    # noisefreq_LS = 1.0

    fitfreq = []

    plaw_pars = []
    plaw_res = []

    brkn_pars = []
    brkn_res = []
    lrntz_pars = []
    lrntz_res = []

    bend_pars = []
    bend_res = []
    
    chi2_all = []
    pvalue_all = []
         
    binstd = []
    mean_arr = []
    
    for file in files:
        
        if file.endswith('_lightcurve_EmanSim_test.dat'):
        
            
            ## Open light curve data and set to dataframe for float parsing
            data = open(os.path.join(subdir, file), "r")
            # pd.set_option('display.float_format','{:.18f}'.format)
            df = pd.read_table(os.path.join(subdir, file), sep = ' ', header=None, dtype='float')
            
            # print(df)
            
            ## Get number of data points in light curve
            datapts = len(data.readlines())


            # time_sec = df[0]
            # LSfreq, LSpow = LombScargle(time_sec, df[1], normalization = 'psd').autopower()
            # print('Poisson noise from LS periodogram is %.10f' %poisson_vdk_LS)
            
            full_data, reduced_ps, poisson = correct_data(df, datapts, subdir, target, num_lcs, bkg, z)
                            
            # reduced_ps, poisson = powerspec(subdir,target,full_data, bkg, mean_arr)
            
            noisefreq  = findnoise(reduced_ps, poisson)

            binnedps, binstd, noisefreq= rebinning(subdir,target,reduced_ps, noisefreq)
            
            # error_df = pd.DataFrame(np.loadtxt(rootdir+target+'/'+target+'_powerr.dat'))
            
            # binstd = find_stdev(error_df)

            fitfreq, plaw_pars, brkn_pars, lrntz_pars, bend_pars, \
                        plaw_res, brkn_res, lrntz_res, bend_res, \
                                chi2_all, pvalue_all = powerlaw_fits(binnedps, noisefreq, binstd)#,np.log10(binstderr)

        

            data.close()

        
    if len(full_data) != 0:  
        
        ## Create figures for newly interpolated fluxes and the normalized flux reduced by a factor of the median

        # fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8,6))
        fig, ax1 = plt.subplots(figsize=(100,8))
        
        ax1.errorbar(full_data[0],full_data[1],yerr = full_data[2],marker='o',markersize=1,color='blue',linestyle='none',alpha=1)

        # ax2.plot(full_data[0],full_data[1]/np.nanmedian(full_data[1]),color='blue',alpha=1)
        # ax2.plot(full_data_pca5[0],full_data_pca5[1]/np.nanmedian(full_data_pca5[1]),color='orange',alpha=0.9)

        
        plt.suptitle(target+': Corrected Light Curve')
        plt.xlabel("Time [s]")
        ax1.set_ylabel("Flux [$e^{-}$ $s^{-1}$]")
        # ax2.set_ylabel("Normalized Flux")
        
        ax1.legend(['Long Simulated Light Curve for '+target])

        plt.savefig(subdir+'/'+target+'_full_simulated_corrected_lc.pdf', format = 'pdf')

        plt.show()
        plt.close()
    

    # if len(LSfreq) != 0 and len(LSfreq_pca5) != 0:
    
    #     ## Create Lomb-Scargle Periodograms with 
        
    #     plt.plot(np.log10(LSfreq), np.log10(LSpow),color='b',alpha=1)
    #     plt.plot(np.log10(LSfreq_pca5), np.log10(LSpow_pca5),color='orange',alpha=0.7)

    #     # plt.hlines(poisson_vdk_LS,0,np.max(LSfreq),colors= 'b',linestyle = 'dotted',label = "Hybrid Poisson Noise")
    #     # plt.hlines(poisson_pca5_vdk_LS,0,np.max(LSfreq_pca5),colors= 'orange',linestyle = (0,(5,10)),label = "PCA5 Poisson Noise") 

    #     # plt.axvline(noisefreq_LS,color= 'b',linestyle = 'dotted',label = "Hybrid noise Frequency")
    #     # plt.axvline(noisefreq_pca5_LS,color= 'orange',linestyle = (0,(5,10)),label = "PCA5 noise Frequency")

    #     plt.title(target+': Lomb-Scargle Periodogram')
    #     plt.xlabel('log('+chr(957)+') [Hz]')
    #     # plt.xlim(left=0)
    #     # plt.ylim(bottom=np.min(np.log10(LSpow))-0.25)
            
    #     plt.ylabel("log(P)")
    #     # plt.ylabel("Power ($e^{y}$) [Units]")
    #     plt.legend(['Hybrid Method','PCA5 Method'])
        
    #     plt.show()
    #     plt.close()

    #     print('Set frequency bounds for fit. Input as log('+chr(957)+') (e.g. for 10e-6 to 10e-5, input -6 then -5).')
    #     low_fbound = float(input('Low bound: '))
    #     high_fbound= float(input('High bound: '))
        

    #     lowf = list(LSfreq).index(find_nearest(LSfreq, np.power(10,low_fbound) ))
    #     highf = list(LSfreq).index(find_nearest(LSfreq, np.power(10,high_fbound) ))
            

    #     LSpars, LScov = curve_fit(f=linear, xdata=np.log10(LSfreq[lowf:highf]), ydata=np.log10(LSpow[lowf:highf]), bounds=(-np.inf, np.inf), maxfev=999999)
    #     LSfitpow = linear(np.log10(LSfreq[lowf:highf]),*LSpars)
    #     LSres = np.log(LSpow[lowf:highf]) - LSfitpow
    #     LSchi2, LSpvalue = stats.chisquare(linear(np.log10(LSfreq[lowf:highf]),*LSpars),f_exp=np.log10(LSpow[lowf:highf]), ddof=len(LSpars),axis = 0)
        

    #     LSpars_pca5, LScov_pca5 = curve_fit(f=linear, xdata=np.log10(LSfreq_pca5[lowf:highf]), ydata=np.log10(LSpow_pca5[lowf:highf]), bounds=(-np.inf, np.inf), maxfev=999999)
    #     LSfitpow_pca5 = linear(np.log10(LSfreq_pca5[lowf:highf]),*LSpars_pca5)
    #     LSres_pca5 = np.log10(LSpow_pca5[lowf:highf]) - LSfitpow_pca5
    #     LSchi2_pca5, LSpvalue_pca5 = stats.chisquare(linear(np.log10(LSfreq_pca5[lowf:highf]),*LSpars_pca5),f_exp=np.log10(LSpow_pca5[lowf:highf]), ddof=len(LSpars_pca5),axis = 0)


    #     fig = plt.figure(figsize=(10,6))
    #     gs = gridspec.GridSpec(nrows=4,ncols= 4,hspace=0.5)
    #     fig.tight_layout()
        
    #     ax0 = fig.add_subplot(gs[:-1, :])
    #     ax0.set_ylabel("log(P)")
    #     ax1 = fig.add_subplot(gs[3,:-1])
    #     ax1.set_xlabel('log('+chr(957)+') [Hz]')
    #     # ax1.set_xlabel('Frequency [Hz]')
    #     ax1.set_ylabel("residuals")
    #     ax2 = fig.add_subplot(gs[3,3])
    #     ax2.axis('off')

    #     ax0.plot(np.log10(LSfreq), np.log10(LSpow),color='b',alpha=1)

    #     alpha = LSpars[0]
    #     ax0.plot(np.log10(LSfreq[lowf:highf]), LSfitpow, color='purple',linestyle='solid',label = chr(945)+'= %.3f'%alpha)

    #     ax1.plot(np.log10(LSfreq[lowf:highf]), LSres,color='purple', label = chr(967)+'$^{2}$: %.2f, p: %.4f'%(LSchi2,LSpvalue)) #, linestyle= 'solid')
    #     ax1.axhline(0, color='k', linestyle = 'solid')

    #     ax0.set_title(target+': Lomb-Scargle Periodogram')
    #     ax1.set_xlabel('log('+chr(957)+') [Hz]')
    #     # plt.xlim(left=0)
    #     # plt.ylim(bottom=np.min(np.log10(LSpow))-0.25)
            
    #     ax0.set_ylabel("log(P)")
    #     ax1.set_ylabel("residuals")

    #     # plt.ylabel("Power ($e^{y}$) [Units]")
        
    #     # if alpha >=1 and alpha_pca5 >= 1:
    #     #   ax0.legend(['Hybrid Method','PCA5 Method',chr(945)+'= %.3f'%alpha,chr(945)+'$_{PCA}$= %.3f'%alpha_pca5])
    #     # if alpha >=1 and alpha_pca5 < 1:
    #     #   ax0.legend(['Hybrid Method','PCA5 Method',chr(945)+'= %.3f'%alpha,chr(945)+'$_{PCA}$= %.3f hr'%(24*alpha_pca5)])
    #     # if alpha < 1 and alpha_pca5 >= 1:
    #     #   ax0.legend(['Hybrid Method','PCA5 Method',chr(945)+'= %.3f hr'%(24*alpha),chr(945)+'$_{PCA}$= %.3f'%alpha_pca5])
    #     # if alpha < 1 and alpha_pca5 < 1:
    #     #   ax0.legend(['Hybrid Method','PCA5 Method',chr(945)+'= %.3f hr'%(24*alpha),chr(945)+'$_{PCA}$= %.3f hr'%(24*alpha_pca5)])

    #     ax0.legend()
    #     ax1.legend(bbox_to_anchor=(1,1))

    #     plt.savefig(subdir+'/'+target+'_corr_sim_LSperiodogram.pdf', format = 'pdf')
        
    #     plt.show()
    #     plt.close()

    if len(reduced_ps) != 0 and len(binnedps) != 0:
    
        for i in range(0,4):
            fig = plt.figure(figsize=(10,6))
            gs = gridspec.GridSpec(nrows=4,ncols= 4,hspace=0.5)
            fig.tight_layout()
            
            ax0 = fig.add_subplot(gs[:-1, :])
            ax0.set_ylabel("log(P)")
            ax1 = fig.add_subplot(gs[3,:-1])
            ax1.set_xlabel('log('+chr(957)+') [Hz]')
            # ax1.set_xlabel('Frequency [Hz]')
            ax1.set_ylabel("residuals")
            ax2 = fig.add_subplot(gs[3,3])
            ax2.axis('off')
            

            # for i in range(1,500):
            #     ax0.scatter(np.log10(reduced_ps[0]), np.log10(reduced_ps[i]), color='b')
            ax0.scatter(np.log10(reduced_ps[0]), np.log10(reduced_ps[1]), color='b')
            ax0.scatter(np.log10(reduced_ps[0]), np.log10(reduced_ps[250]), color='b')
            ax0.scatter(np.log10(reduced_ps[0]), np.log10(reduced_ps[500]), color='b')
        
            ax0.errorbar(np.log10(binnedps[0]), np.log10(binnedps[1]), yerr = np.log10(binstd), color='k', linestyle= 'solid', label = 'Binned PS')
            
            # ax0.scatter(np.log10(binnedps[0]), np.log10(binnedps[1]), s=55, color='k', marker= '|')
            
            ax0.axhline(np.log10(poisson), color= 'b',linestyle = 'dotted') #, label = "Hybrid Poisson Noise")

            ax0.axvline(np.log10(noisefreq), color= 'b',linestyle = 'dotted') #,label = chr(957)+'$_{b}$ ~ %.2e Hz'%noisefreq)

            # ax0.plot(fitfreq, power_law(fitfreq, pars[0],pars[1]),color='purple', linestyle= 'solid')
            # ax0.plot(fitfreq_pca5, power_law(fitfreq_pca5, pars_pca5[0],pars_pca5[1]),color='g', linestyle= 'dashed')
        
            if i == 0:
                ax0.plot(fitfreq, linear(fitfreq, *plaw_pars),color='purple', linestyle= 'solid', label = chr(945)+' = %.3f'%plaw_pars[0])
            
                ax1.scatter(fitfreq, plaw_res,s=100,color='purple',marker='_', label = chr(967)+'$^{2}$: %.2f, p: %.4f'%(chi2_all[0],pvalue_all[0])) #, linestyle= 'solid')
                ax1.axhline(0, color='k', linestyle = 'solid')
                
                ax1.set_ylim(-1.5*np.max(plaw_res),1.5*np.max(plaw_res))
                ax0.set_title("Mean-subtracted flux: single p-law fit")

            elif i == 1:
                # ax0.plot(fitfreq, piecewise_linear(fitfreq, *pars),color='purple', linestyle= 'solid', label = chr(945)+'= %.3f'%pars[2])
                breakfreq = find_intersection(*brkn_pars[1:])
                # turn_freq = fitfreq.index(find_nearest(fitfreq,pars[0]))
                lowf = np.linspace(fitfreq[0],breakfreq,num=25)
                highf = np.linspace(breakfreq,fitfreq[-1:],num=25)
                
                ax0.plot(lowf, linear(lowf, brkn_pars[1], brkn_pars[3]),color='purple', linestyle= 'solid')
                ax0.plot(highf, linear(highf, brkn_pars[2], brkn_pars[4]),color='purple', linestyle= 'solid', label = chr(945)+'= %.3f'%brkn_pars[2])
                ax0.axvline(breakfreq,color= 'purple',linestyle = 'dotted', label = chr(957)+'$_{b}$ ~ 10$^{%.2f}$ Hz'%breakfreq)    
        

                ax1.scatter(fitfreq, brkn_res,s=100,color='purple',marker='_', label = chr(967)+'$^{2}$: %.2f, p: %.4f'%(chi2_all[1],pvalue_all[1])) #, linestyle= 'solid')
                ax1.axhline(0, color='k', linestyle = 'solid')
                
                ax1.set_ylim(-1.5*np.max(brkn_res),1.5*np.max(brkn_res))
                ax0.set_title("Mean-subtracted flux: broken p-law fit")

            elif i == 2:

                center = lrntz_pars[0]
                ax0.plot(fitfreq, combolorentz(fitfreq, *lrntz_pars),color='purple', linestyle= 'solid', label = chr(945)+'= %.3f'%lrntz_pars[2])
                ax0.axvline(center,color= 'purple',linestyle = 'dotted', label = chr(957)+'$_{c}$ ~ 10$^{%.2f}$ Hz'%center)
                
                ax1.scatter(fitfreq, lrntz_res,s=100,color='purple',marker='_', label = chr(967)+'$^{2}$: %.2f, p: %.4f'%(chi2_all[2],pvalue_all[2])) #, linestyle= 'solid')
                ax1.axhline(0, color='k', linestyle = 'solid')
                
                ax1.set_ylim(-1.5*np.max(lrntz_res),1.5*np.max(lrntz_res))
                ax0.set_title("Mean-subtracted flux: Lorentzian fit")

            elif i == 3:

                knee = np.log10(bend_pars[0])
                ax0.plot(fitfreq, linearknee(np.power(10,fitfreq), *bend_pars),color='purple', linestyle= 'solid', label = chr(945)+'= %.3f'%bend_pars[1])
                ax0.axvline(knee,color= 'purple',linestyle = 'dotted', label = chr(957)+'$_{k}$ ~ 10$^{%.2f}$ Hz'%knee)
                # print(len(fitfreq))
                # print(len(bend_res))

                ax1.scatter(fitfreq, bend_res,s=100,color='purple',marker='_', label = chr(967)+'$^{2}$: %.2f, p: %.4f'%(chi2_all[3],pvalue_all[3])) #, linestyle= 'solid')
                ax1.axhline(0, color='k', linestyle = 'solid')
                
                ax1.set_ylim(-1.5*np.max(bend_res),1.5*np.max(bend_res))
                ax0.set_title("Mean-subtracted flux: Knee-bend fit")
        
            plt.suptitle(target+': Power Spectrum')
            
            ymax = np.max(np.log10(reduced_ps[1]))
            

            ymin = np.min(np.log10(reduced_ps[1]))
        
            ax0.set_ylim(ymin-0.5,ymax+0.5)
            ax0.set_xlim(np.min(np.log10(reduced_ps[0]))-0.2,np.max(np.log10(reduced_ps[0]))+0.2)
            ax1.set_xlim(left = np.min(np.log10(reduced_ps[0]))-0.2)
            
            ax0.legend(fontsize = 'small', loc = 'best')
        
            ax1.legend(bbox_to_anchor=(1,1))

            if i == 0:
                plt.savefig(subdir+'/'+target+'_full_corr_sim_Powerspec_singleplaw.pdf', format = 'pdf')
            elif i == 1:
                plt.savefig(subdir+'/'+target+'_full_corr_sim_Powerspec_brknplaw.pdf', format = 'pdf')
            elif i == 2:
                plt.savefig(subdir+'/'+target+'_full_corr_sim_Powerspec_lorentzian.pdf', format = 'pdf')
            elif i == 3:
                plt.savefig(subdir+'/'+target+'_full_corr_sim_Powerspec_kneebend.pdf', format = 'pdf')
                
            plt.show()
            plt.close()


    if count == 2:
        sys.exit()