#### ---- Simple Lake Model Using Mass Balance ---- ####

# Written by P. Saara Ngom 2025 and edited by Juan P. Sierra

# -- Libraries -- #

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import rasterio
from matplotlib.lines import Line2D
from datetime import datetime


# -- Supporting functions -- ##

def remplissage_lac(mnt_or_bathy_path,seuil_alt_niveau_eau):
    """ input:  - Bathimetry path
                - a list with different altitudes of water surface

        return: - 1 list with water level
                - 1 list with water volume linked to water levels (i.e. volume between lake bottom and the water level altitude in the raster)
                - 1 list with water surfaces linked to different water levels """

    #varable de sockage
    volume_stock=[]
    surface_stock=[]
    altitude_stock=[]

    for i in seuil_alt_niveau_eau:
        # === OUVERTURE DU RASTER ===
        with rasterio.open(mnt_or_bathy_path) as src:
            mnt = src.read(1)  # lecture de la couche raster (1 seule couche ici)
            #profil = src.profile
            transform = src.transform
            #nodata=src.nodata

            # Récupération de la taille d'un pixel en mètres (résolution spatiale)
            pixel_size_x = transform[0]
            pixel_size_y = -transform[4]  # négatif car dans le raster les Y décroissent
            pixel_area = pixel_size_x * pixel_size_y

            # Création d’un masque des pixels sous le seuil
            masque = (mnt < i) & (mnt > 0)

            # Calcul des hauteurs d’eau (seuil - altitude du sol), uniquement pour les pixels sous le seuil
            hauteurs = np.where(masque, i - mnt, 0)

            # === CALCULS ===
            surface_eau = np.sum(masque) * pixel_area  # en m²
            volume = np.sum(hauteurs) * pixel_area     # en m³

            surface_stock.append(surface_eau)
            volume_stock.append(volume)
            altitude_stock.append(i)
    return altitude_stock,surface_stock,volume_stock

def relation_volume_en_surface(data_volume,data_surface):
    """ input: list of water volumes and water surfaces corresponding to the output of the function "remplissage_lac".
        return: the relationship that allows to pass from volumes to surfaces"""

    S_interpolation= interp1d(
        data_volume,
        data_surface,
        kind="linear",
        fill_value='extrapolate')  # <-- active l'extrapolation
    return S_interpolation

def relation_volume_en_hauteur(data_volume,data_hauteur):
    """ input: liste des hauteur d'eau et des surfaces correspondant au return de la fonction "remplissage_lac".
        return: la relation permettant de passer des volumes aux hauteurs"""

    h_interpolation= interp1d(
        data_volume,
        data_hauteur,
        kind='linear',
        fill_value='extrapolate')  # <-- active l'extrapolation
    return h_interpolation

def relation_surface_en_volume(data_volume,data_surface):
    """ input: list of water volumes and water surfaces corresponding to the output of the function "remplissage_lac".
        return: the relationship that allows to pass from surfaces to volumes"""

    S_interpolation= interp1d(
        data_surface,
        data_volume,
        kind="linear",
        fill_value='extrapolate')  # <-- active l'extrapolation
    return S_interpolation

# -- Main function -- #

def model(bathy_path_abbe,bathy_path_lakeN,df_data,col_debit,col_precip,col_evap,a,b,h_ini_abbe=0,V_ini_abbe=0,S_ini_abbe=0,h_ini_lakeN=0,V_ini_lakeN=0,S_ini_lakeN=0,seuil_alt_niveau_eau_abbe=np.linspace(175, 330,100),seuil_alt_niveau_eau_lakeN=np.linspace(175, 330,100),additional_inputs_abbe=None):
    """ input:  -mnt_or_bathy_path: path of raster bathimetry of lakes Abbe and northern lakes Afambo and Gemeri
                -df_data: dataframe with necessary inputs, 1 line = 1 month.
                -col_debit,col_precip,col_evap: columns with stream flow (m3), rainfall (mm) and evapotranspiration (mm) dataframe "df_data'
                -h_ini,S_ini,V_ini: the initial height level (m), initial surface (m2), and initial volume (m3) given by the function "remplissage_lac".
                -seuil_alt_niveau_eau: water levels for which the relationship volume/sirface/height will be built. Take the minimum altitude as the lake depth.
                -a,b: model parameters (float) between 0 and 1
                -additional_inputs_abbe (list of column names)
        return: -a dataframe with the volume computed using the mass balance, the modeled height, modeled surface at monthly resolution"""

    ### --- Building the volume/surface relationship --- ###

    # -- Lake Abbe -- ##
    h_etablir_relation_abbe,S_etablir_relation_abbe,V_etablir_relation_abbe=remplissage_lac(bathy_path_abbe,seuil_alt_niveau_eau_abbe)
    f_VS_abbe=relation_volume_en_surface(V_etablir_relation_abbe,S_etablir_relation_abbe)
    f_Vh_abbe=relation_volume_en_hauteur(V_etablir_relation_abbe,h_etablir_relation_abbe)
    # -- Northern Lakes -- #
    h_etablir_relation_lakeN,S_etablir_relation_lakeN,V_etablir_relation_lakeN=remplissage_lac(bathy_path_lakeN,seuil_alt_niveau_eau_lakeN)
    f_VS_lakeN=relation_volume_en_surface(V_etablir_relation_lakeN,S_etablir_relation_lakeN)
    f_Vh_lakeN=relation_volume_en_hauteur(V_etablir_relation_lakeN,h_etablir_relation_lakeN)

    ### --- Build up the ouput dataframe for results --- ###

    df_model = df_data.copy()
    df_model['model_cumul precip sur abbe (m3)'] = np.nan
    df_model['model_cumul precip sur lakeN (m3)'] = np.nan
    df_model['model_evap_abbe (m3)'] = np.nan
    df_model['model_evap_lakeN (m3)'] = np.nan
    df_model['model_debit lacs (m3)'] = np.nan #debit atteignant les lacs (Q0 - prélèvemet)
    df_model['model_debit direct Abbe (m3)'] = np.nan #debit Q1 sur le schéma (C.f page 12 du rapport de stage)
    df_model['model_debit lacs nord (m3)'] = np.nan #debit Q2 (C.f page 12 du rapport de stage))
    df_model['model_Volume_abbe (m3)']=np.nan
    df_model['model_Volume_lakeN (m3)']=np.nan
    df_model['model_Surface_abbe (m2)']=np.nan
    df_model['model_Surface_lakeN (m2)']=np.nan
    df_model['model_surface_total (m2)']=np.nan
    df_model['model_hauteur_abbe (m)']=np.nan
    df_model['model_hauteur_lakeN (m)']=np.nan
    df_model['diff_obsv_model_abbe (m2)']=np.nan
    df_model['diff_obsv_model_lakeN (m2)']=np.nan

    ### --- Initialisation --- ###

    V_abbe=V_ini_abbe
    S_abbe=S_ini_abbe
    h_abbe=h_ini_abbe

    V_lakeN=V_ini_lakeN
    S_lakeN=S_ini_lakeN
    h_lakeN=h_ini_lakeN

    ### --- Parameterization --- ### How these values are found?
    p=1-a-b
    X= a/(1-p)
    Y=1-X

    ### --- Computation of the streamflows arriving to each lake --- ###
    for index, debit, precip, evap in zip(df_model.index, df_model[col_debit],df_model[col_precip], df_model[col_evap]):
        #Callage des paramètres
        #debit_lacs=debit*(1-p)
        #debit_direct_abbe=debit*a
        #debit_lacs_nord=debit*b

        debit_lacs=debit*(1-p) # streamflow arriving to the lakes
        debit_direct_abbe=debit*X*(1-p) # portion of the Awash stramflow flowing to the lake Abbe
        debit_lacs_nord=debit*Y*(1-p) # portion of the Awash stramflow flowing to north
        diff_model_obsv_abbe=df_model.loc[index,'Surf_water_abbe (km2)']*10**6 - S_abbe
        diff_model_obsv_lakeN=df_model.loc[index,'Surf_water_lakeN (km2)']*10**6 - S_lakeN

        df_model.loc[index,'model_debit lacs (m3)'] = debit_lacs
        df_model.loc[index,'model_debit direct Abbe (m3)'] = debit_direct_abbe
        df_model.loc[index,'model_debit lacs nord (m3)'] = debit_lacs_nord
        df_model.loc[index,'model_cumul precip sur lakeN (m3)'] = (precip/1000)*(S_lakeN)

        df_model.loc[index,'model_cumul precip sur abbe (m3)'] = (precip/1000)*(S_abbe)
        df_model.loc[index,'model_evap_abbe (m3)'] = (evap/1000)*S_abbe
        df_model.loc[index,'model_evap_lakeN (m3)'] = (evap/1000)*S_lakeN

        df_model.loc[index,'model_Volume_abbe (m3)']=V_abbe
        df_model.loc[index,'model_Volume_lakeN (m3)']=V_lakeN
        df_model.loc[index,'model_Surface_abbe (m2)']=S_abbe
        df_model.loc[index,'model_Surface_lakeN (m2)']=S_lakeN
        df_model.loc[index,'model_surface_total (m2)']=S_abbe+S_lakeN
        df_model.loc[index,'model_hauteur_abbe (m)']=h_abbe
        df_model.loc[index,'model_hauteur_lakeN (m)']=h_lakeN
        df_model.loc[index,'diff_obsv_model_abbe (m2)']= diff_model_obsv_abbe
        df_model.loc[index,'diff_obsv_model_lakeN (m2)']= diff_model_obsv_lakeN

        ### --- Mass balance --- ###

        V_abbe= V_abbe - df_model.loc[index,'model_evap_abbe (m3)'] + debit_direct_abbe + debit_lacs_nord - df_model.loc[index,'model_evap_lakeN (m3)'] + df_model.loc[index,'model_cumul precip sur abbe (m3)'] + df_model.loc[index,'model_cumul precip sur lakeN (m3)'] # It is missing the rainfall over northern lakes, right?
        V_lakeN= V_lakeN - df_model.loc[index,'model_evap_lakeN (m3)'] + debit_lacs_nord + df_model.loc[index,'model_cumul precip sur lakeN (m3)']

        if additional_inputs_abbe:
            for col in additional_inputs_abbe:
                V_abbe += df_model.loc[index, col]

        ### --- Volume -> Surface relationship --- ###
        S_abbe = f_VS_abbe(V_abbe)
        S_lakeN = f_VS_lakeN(V_lakeN)

        ### --- Volume -> water level relationship --- ###
        h_abbe = f_Vh_abbe(V_abbe)
        h_lakeN=f_Vh_lakeN(V_lakeN)

    ### --- Computing RMSE --- ###
    mse_abbe = np.sum((df_model['diff_obsv_model_abbe (m2)'].apply(lambda x:x**2)))
    rmse_abbe = np.sqrt(mse_abbe/len(df_model['diff_obsv_model_lakeN (m2)']))
    mse_lakeN = np.sum((df_model['diff_obsv_model_lakeN (m2)'].apply(lambda x:x**2)))
    rmse_lakeN = np.sqrt(mse_lakeN/len(df_model['diff_obsv_model_lakeN (m2)']))

    return df_model,h_etablir_relation_abbe,S_etablir_relation_abbe,V_etablir_relation_abbe,h_etablir_relation_lakeN,S_etablir_relation_lakeN,V_etablir_relation_lakeN,rmse_abbe,rmse_lakeN


