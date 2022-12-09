from plots_hists_angle import loss_plot, paz_angle_hist, p_angle_hist
from resolution import res_plot, lnE_res ,E_res

def paz_show_info(path_start, big_list):
    loss_lists, polar_error, azimut_error, res_hists, polar_hists, azimut_hists = big_list

    loss_plot(list_test = loss_lists[1], list_train = loss_lists[0], 
                  path = path_start + "/Plots/Loss.png" )
              
    res_plot(polar_error[0], polar_error[1],  #hist_train_polar_error, hist_val_polar_error,
                 path = path_start + "/Hists/Polar_Error.png", res_name = "Polar_Error ")
        
    #histogram of azimut angle errors"
    res_plot(azimut_error[0], azimut_error[1],  #hist_train_azimut_error, hist_val_azimut_error",
                 path = path_start + "/Hists/Azimut_Error.png", res_name = "Azimut_Error ") 
        
    #histogram of polar angle errors for specific angles",
    res_plot(polar_error[2], polar_error[3], #hist_train_polar_error_angle_cut, hist_val_polar_error_angle_cut,"
                 path = path_start + "/Hists/Polar_Error_Angle_Cut.png", res_name = "Polar_Error_Angle_Cut ") 

    #resolution histograms",
    res_plot(res_hists[0],res_hists[1],
                 path = path_start + "/Hists/RESOLUTIONS.png", res_name = "Resolution ")
        
    #resolution histograms for specific angles",
    res_plot(res_hists[2], res_hists[3],
                 path = path_start + "/Hists/Resolution_Angle_Cut.png", res_name = "Resolution_Angle_Cut ")
        
    #angle histograms",
    paz_angle_hist(polar_hists[0], azimut_hists[0], name='train', path = path_start + "/Hists/Angles_distr_train.png")
    paz_angle_hist(polar_hists[1], azimut_hists[1], name='val', path = path_start + "/Hists/Angles_distr_val.png")
    
    
    
    
def p_show_info(path_start, big_list):
    loss_lists, polar_error, polar_hists = big_list

    loss_plot(list_test = loss_lists[1], list_train = loss_lists[0],  path = path_start + "/Plots/LOSS.png")
        
    #histogram of polar angle errors
    res_plot(polar_error[0], polar_error[1],  #hist_train_polar_error, hist_val_polar_error
                 path = path_start + "/Hists/Polar_Error.png", res_name = "Polar_Error")  

    #histogram of polar angle errors for specific angles
    res_plot(polar_error[2], polar_error[3], #hist_train_polar_error_angle_cut, hist_val_polar_error_angle_cut,  
             path = path_start + "/Hists/Polar_Error_Angle_Cut.png", res_name = "Polar_Error_Angle_Cut ") 

    #angle histograms
    p_angle_hist(polar_hists[0], name='train', path = path_start + "/Hists/Angles_train.png")
    p_angle_hist(polar_hists[1], name='val', path = path_start + "/Hists/Angles_val.png")  
    
    loss_lists = [loss_train , loss_test]
    polar_hists = [hist_train_polar , hist_val_polar]
    polar_error = [hist_train_polar_error, hist_val_polar_error,
                       hist_train_polar_error_angle_cut, hist_val_polar_error_angle_cut]   
    
def lnE_show_info(path_start, loss_lists, lnE_error):
    #loss plot
    loss_plot(list_test = loss_lists[1], list_train = loss_lists[0],  path = path_start + "/Plots/LOSS.png" )
        
    #histogram of lnE errors
    res_plot(lnE_error[0], lnE_error[1],  #hist_train_lnE_error, hist_val_lnE_error,
                 path = path_start + "/Hists/lnE_Error.png", res_name = "lnE_Error ")    
