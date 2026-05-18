#### --- Script for finding the lake model dynamic parameters 1985-2024 ---- ####
# Written by Juan P. Sierra 2025 based on the original scripts from P. Saara Ngom

### -- Libraries -- ###

import pickle
import copy
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import rasterio
from matplotlib.lines import Line2D
from datetime import datetime
from Lake_Model_Functions import remplissage_lac, relation_volume_en_surface, model, relation_surface_en_volume
import seaborn as sns
import os
import matplotlib.cm as cm

#### ---- Functions ---- ####

def fill_nan_with_avg(df, col):
    """ input: dataframe and the name of the interesting column
        return: the dataframe without NaN (using the average of the neighbor data before and after the NaN)"""
    series = df[col]
    filled = series.copy()

    for i in range(len(series)):
        if pd.isna(series.iloc[i]):
            current_idx = series.index[i]

            # Cherche l'index précédent avec une valeur non-NaN
            prev_idx = series.loc[:current_idx].last_valid_index()
            # Cherche l'index suivant avec une valeur non-NaN
            next_idx = series.loc[current_idx:].first_valid_index()

            if prev_idx is not None and next_idx is not None:
                avg = (series.loc[prev_idx] + series.loc[next_idx]) / 2
                filled.iloc[i] = avg
    df[col] = filled
    return df

def fill_with_climatology(x):
    if pd.isna(x):
        return climatology[x.name.month]
    else:
        return x


#### --- Preparation of the model inputs ---- ####

## -- Bathimetry -- ##

MNT_path_abbe_1="/scratchx/jpsierra/NILAFAR/Lake_Model/interpol_abbe_GOCAD_1_alt_min_156m.tif"  #alt min = 156m

MNT_path_abbe_2="/scratchx/jpsierra/NILAFAR/Lake_Model/interpol_abbe_GOCAD_2_alt_min_188m.tif"

MNT_path_abbe_3="/scratchx/jpsierra/NILAFAR/Lake_Model/interpol_abbe_GOCAD_3_alt_min_205m.tif"

MNT_path_lakeN="/scratchx/jpsierra/NILAFAR/Lake_Model/interpol_Gemeri_Afambo_GOCAD.tif"

## -- Streamflow from GRDC -- ##

grdc1 = pd.read_csv("/scratchx/jpsierra/NILAFAR/Lake_Model/1577603_GRDC_Qd.txt", sep=';', engine='python')

grdc1['Date'] = pd.to_datetime(grdc1['Date'])
grdc1 = grdc1.set_index('Date')
grdc1 = grdc1.resample('M').mean()
grdc1["Qm3_total"] = grdc1["Qm3s"] *60*60*24*grdc1.index.days_in_month

# Filling missing data with monthly climatology #

q = grdc1["Qm3_total"]
q_filled = q.fillna(q.groupby(q.index.month).transform('mean'))

grdc1["Qm3_total_filled"] = q_filled

## -- Streamflow from GR2M -- ##

data_debit = pd.read_csv(
    "/scratchx/jpsierra/NILAFAR/Lake_Model/BV_GRDC__GLEAM_CHIRPS__GR2M_QSIM_1985-2024.txt",
    sep=";",
    index_col=0,
    header=0
)

data_debit.index=pd.to_datetime(data_debit.index)
data_debit.head()
#data_debit=data_debit[['1577603']] #BV correspondant à celui de tendaho
data_debit=data_debit[['1577100']] #BV correspondant à celui de tendaho
Surf_Tendaho= 62088*10**6 #m2: surface du BV de Tendaho (GRDC)
data_debit=data_debit.apply(lambda x: (x/1000)*Surf_Tendaho) #/1000 car data_debit est en mm
data_debit.columns=["Q_TENDAHO (m3)"]

# -- Increasing above percentile 90 values -- #

p90 = data_debit['Q_TENDAHO (m3)'].quantile(0.90)
mask_p90 = data_debit['Q_TENDAHO (m3)'] > p90
indices_above_p90 = data_debit.index[mask_p90]

data_debitp90 = data_debit.copy()
data_debitp90.loc[mask_p90, 'Q_TENDAHO (m3)'] *= 2

## Zhao lakes ##

file_path = "/scratchx/jpsierra/NILAFAR/Lake_Model/1_openwater_area.csv"
df2 = pd.read_csv(file_path) # Open water area m**2
lake_codes = [1548,15810,15812]
lake_names = ["Abbe","Gemeri","Afambo"]
lakes_opw_area = df2[df2['Hylak_id'].isin(lake_codes)]
del df2
df2 = lakes_opw_area.set_index('Hylak_id')
df2_transposed = df2.T
df2_transposed.index = pd.to_datetime(df2_transposed.index)
lake_codes_int = list(map(int, lake_codes))
id_to_name = dict(zip(lake_codes_int, lake_names))
opw_area_dict = {
    id_to_name[hylak_id]: df2_transposed[hylak_id]
     for hylak_id in df2_transposed.columns if hylak_id in id_to_name
}

## -- Observed lakes from Peckel -- ##

# - All lakes - #

df = pd.read_csv("/scratchx/jpsierra/NILAFAR/Lake_Model/surface_water_timeseries.csv")
df = df.rename(columns={"Unnamed: 0": "Date"})
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# - Lake Abbe - #

data_surf_lac_abbe = df['abbe_pekel']
data_surf_lac_abbe = data_surf_lac_abbe.to_frame(name="Surf_water (km2)")

# - Lake Gemeri - #

data_surf_lac_Gemeri=df['gemeri_pekel']
data_surf_lac_Gemeri = data_surf_lac_Gemeri.to_frame(name="Surf_water (km2)")

# - Lake Afambo - #

data_surf_lac_Afambo=df['afambo_pekel']
data_surf_lac_Afambo = data_surf_lac_Afambo.to_frame(name="Surf_water (km2)")

# - Lake Bari - #

data_surf_lac_Bari=df['bari_pekel']
data_surf_lac_Bari = data_surf_lac_Bari.to_frame(name="Surf_water (km2)")

# - Northern lakes - #

S_lakeN_nan=data_surf_lac_Gemeri+data_surf_lac_Afambo

## -- Observed lakes from Hydroweb -- ##

hydroweb=pd.read_csv("/scratchx/jpsierra/NILAFAR/Lake_Model/L_Abhe_hauteur_surface_volume_HYDROWEB.csv",sep=';',index_col=0,header=0) ### !!! Is this the correct file?
hydroweb.index=pd.to_datetime(hydroweb.index, format="%d/%m/%Y %H:%M")

start = hydroweb.index.min() #date début des données hydroweb
end = hydroweb.index.max()  #date fin des données hydroweb
firsts = pd.date_range(start, end, freq='MS')  #format "YYYY-MM-01"

# interpolation pour avoir une hauteur d'eau au début de chaque mois
df_interp=hydroweb[['L_abhe - Water Surface Elevation - values(m)']].copy()
df_interp = df_interp.reindex(hydroweb.index.union(firsts))  # ajoute les 1ers de mois
df_interp =df_interp.sort_index().interpolate(method='time') #interpol linéairement ### !!! I am not sure this is appropriate
df_interp.rename(columns={"L_abhe - Water Surface Elevation - values(m)":"h water level (m)"},inplace=True)
data_h_level= df_interp[df_interp.index.day == 1] #pour séléectionner uniquement les valeurs de début du mois

## -- Precipitation from CHIRPSv2 -- ##

data_pluie_CHIRPS_mois = xr.open_dataset('/scratchx/jpsierra/NILAFAR/Lake_Model/CHIRPSv2_MonthlyRainfall_Lakes.nc')

data_pluie_CHIRPS_mois_abbe = data_pluie_CHIRPS_mois["precip"].sel(lake_id=1548)
data_pluie_CHIRPS_mois_abbe = data_pluie_CHIRPS_mois_abbe.to_series()
data_pluie_CHIRPS_mois_abbe.name = "BV_Abbe"
data_pluie_CHIRPS_mois_abbe.index = data_pluie_CHIRPS_mois_abbe.index.to_period("M").start_time

## -- Evaporation data from GLEAMv42a -- ##

data_evap_gleam = xr.open_dataset('/scratchx/jpsierra/NILAFAR/Lake_Model/GLEAM_MonthlyPotentialEvap_Lakes.nc')

data_evap_gleam_abbe = data_evap_gleam["Ep"].sel(lake_id=1548)
data_evap_gleam_abbe = data_evap_gleam_abbe.to_series()
data_evap_gleam_abbe.name = "Evap moy (mm)"
data_evap_gleam_abbe.index = data_evap_gleam_abbe.index.to_period("M").start_time

date =  opw_area_dict['Abbe'].loc['1985-01-01':'1985-01-01'].index[0]

date_debut=date
date_only = date.strftime('%Y-%m-%d')
date_fin='2024-12-01'
date_fin = pd.Timestamp(date_fin)

date_debut = pd.Timestamp('1985-01-01')
date_fin   = pd.Timestamp('2024-12-01')

windows = []

start = date_debut
while start + pd.DateOffset(years=10) <= date_fin:
    end = start + pd.DateOffset(years=10)
    windows.append((start, end))
    start = start + pd.DateOffset(years=1)

print("SCRIPT STARTED", flush=True)

results_windows = {}
for i,time_window in enumerate(windows):    #Loop over 10-year time windows
    best_rmse = np.inf
    best_a = None
    best_b = None
    date_st =  time_window[0]
    date_en = time_window[1]
    surf_abbe = data_surf_lac_abbe.loc[date_st:date_en]
    surf_abbe = surf_abbe['Surf_water (km2)']
    surf_abbe.name = 'Surface Water'
    climatology = surf_abbe.groupby(surf_abbe.index.month).mean()
    surf_abbe_filled = surf_abbe.copy()
    mask = surf_abbe.isna()
    surf_abbe_filled.loc[mask] = surf_abbe.index[mask].month.map(climatology)

    surf_nlakes = S_lakeN_nan.loc[date_st:date_en]
    surf_nlakes = surf_nlakes['Surf_water (km2)']
    surf_nlakes.name = 'Surface Water'
    climatology = surf_nlakes.groupby(surf_nlakes.index.month).mean()
    surf_nlakes_filled = surf_nlakes.copy()
    mask = surf_nlakes.isna()
    surf_nlakes_filled.loc[mask] = surf_nlakes.index[mask].month.map(climatology)

    E = data_evap_gleam_abbe[date_st:date_en]
    E=E*1.5
    P = data_pluie_CHIRPS_mois_abbe[date_st:date_en]
    Q=data_debitp90["Q_TENDAHO (m3)"][date_st:date_en]
    Q.index = Q.index.to_period('M').to_timestamp('D')
    S=surf_nlakes_filled + surf_abbe_filled
    S_abbe=surf_abbe_filled
    S_lakeN=surf_nlakes_filled
    h=data_h_level['h water level (m)'][date_st:date_en]

    df_data_compile=pd.concat([S,S_abbe,S_lakeN,Q,h,P,E],axis=1) #concaténation
    df_data_compile.columns=['Surf_water (km2)','Surf_water_abbe (km2)','Surf_water_lakeN (km2)', 'Q_TENDAHO (m3)','h water level (m)','flux precip lacs (mm)', 'flux evap lacs (mm)']

    df_data=df_data_compile.copy()
    df_data=fill_nan_with_avg(df_data,'Surf_water (km2)')
    df_data=fill_nan_with_avg(df_data,'Surf_water_lakeN (km2)')
    df_data=fill_nan_with_avg(df_data,'Surf_water_abbe (km2)')

    closest_date = opw_area_dict['Abbe'].index.get_indexer([date_st], method='nearest')

    S_ini_abbe = opw_area_dict['Abbe'].iloc[closest_date[0]]
    S_ini_lakeN = opw_area_dict['Gemeri'].iloc[closest_date[0]] + opw_area_dict['Afambo'].iloc[closest_date[0]]

    if i==0:
        seuil_alt_niveau_eau_abbe=np.linspace(152, 252,110)
        seuil_alt_niveau_eau_lakeN=np.linspace(330, 350,20)
        bathy_path_abbe = MNT_path_abbe_3
        bathy_path_lakeN = MNT_path_lakeN
        h_etablir_relation_abbe,S_etablir_relation_abbe,V_etablir_relation_abbe=remplissage_lac(bathy_path_abbe,seuil_alt_niveau_eau_abbe)

        S_array = np.array(S_etablir_relation_abbe)
        V_array = np.array(V_etablir_relation_abbe)
        h_array = np.array(h_etablir_relation_abbe)
        S_abbe = copy.deepcopy(S_array)
        V_abbe = copy.deepcopy(V_array)

        sort_idx = np.argsort(S_array)
        S_sorted = S_array[sort_idx]
        V_sorted = V_array[sort_idx]
        h_sorted = h_array[sort_idx]

        h_ini_abbe = np.interp(S_ini_abbe, S_sorted, h_sorted)
        V_ini_abbe = np.interp(S_ini_abbe, S_sorted, V_sorted)

        h_etablir_relation_lakeN,S_etablir_relation_lakeN,V_etablir_relation_lakeN=remplissage_lac(bathy_path_lakeN,seuil_alt_niveau_eau_lakeN)

        S_array = np.array(S_etablir_relation_lakeN)
        V_array = np.array(V_etablir_relation_lakeN)
        h_array = np.array(h_etablir_relation_lakeN)

        sort_idx = np.argsort(S_array)
        S_sorted = S_array[sort_idx]
        V_sorted = V_array[sort_idx]
        h_sorted = h_array[sort_idx]

        h_ini_lakeN = np.interp(S_ini_lakeN, S_sorted, h_sorted)
        V_ini_lakeN = np.interp(S_ini_lakeN, S_sorted, V_sorted)
        param_values = np.arange(0.05, 0.5001, 0.02)

    for a_orig in param_values:
        max_b = 0.5 - a_orig
        b_values = np.arange(0.05, max_b, 0.02)
        for b_orig in b_values:
            df_model_orig,h_etablir_relation_abbe,S_etablir_relation_abbe,V_etablir_relation_abbe,h_etablir_relation_lakeN,S_etablir_relation_lakeN,V_etablir_relation_lakeN,rmse_abbe,rmse_lakeN=model(MNT_path_abbe_3,MNT_path_lakeN,df_data,"Q_TENDAHO (m3)","flux precip lacs (mm)","flux evap lacs (mm)",a_orig,b_orig,h_ini_abbe=h_ini_abbe,S_ini_abbe=S_ini_abbe,V_ini_abbe=V_ini_abbe,h_ini_lakeN=h_ini_lakeN,S_ini_lakeN=S_ini_lakeN,V_ini_lakeN=V_ini_lakeN,seuil_alt_niveau_eau_abbe=np.linspace(152, 252,110),seuil_alt_niveau_eau_lakeN=np.linspace(330, 350,20))
            if rmse_abbe < best_rmse:
                best_rmse = rmse_abbe
                best_a = a_orig
                best_b = b_orig
    results_windows[(date_st, date_en)] = {
        "a_best": best_a,
        "b_best": best_b,
        "rmse_abbe_min": best_rmse
    }
    df_results = pd.DataFrame([
        {
            "start": k[0],
            "end": k[1],
            "a_best": v["a_best"],
            "b_best": v["b_best"],
            "rmse_abbe_min": v["rmse_abbe_min"]
        }
        for k, v in results_windows.items()
    ])
    print(df_results)

windows = []
time_start = []
time_end = []
a_vals = []
b_vals = []
rmse_vals = []

for i, ((t0, t1), vals) in enumerate(results_windows.items()):
    windows.append(i)
    time_start.append(pd.Timestamp(t0))
    time_end.append(pd.Timestamp(t1))
    a_vals.append(vals['a_best'])
    b_vals.append(vals['b_best'])
    rmse_vals.append(vals['rmse_abbe_min'])

ds = xr.Dataset(
    {
        "a_best": ("window", a_vals),
        "b_best": ("window", b_vals),
        "rmse_abbe_min": ("window", rmse_vals),
    },
    coords={
        "window": windows,
        "time_start": ("window", time_start),
        "time_end": ("window", time_end),
    }
)

ds.to_netcdf("/scratchx/jpsierra/NILAFAR/Lake_Model/decadal_parameter_results_GR2M-GLEAM-CHIRPS.nc")

