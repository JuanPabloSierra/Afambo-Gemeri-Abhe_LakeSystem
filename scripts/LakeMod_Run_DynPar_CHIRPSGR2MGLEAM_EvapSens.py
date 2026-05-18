#### --- Script for running the model in the period 1985-2024 ---- ####
# Written by Juan P. Sierra based on the original code by P. Saara Ngom 2025
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

## -- Model Parameters -- ##

m_param = xr.open_dataset('/scratchx/jpsierra/NILAFAR/Lake_Model/decadal_parameter_results.nc')

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

## -- Precipitation from CHIRPSv2-- ##

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

## -- Preparing the input dataframe -- ##

dates_loop = opw_area_dict['Abbe'].loc['1985-01-01':'1985-01-01'].index
results_abbe = {}
results_nlakes = {}

evap_factors = np.arange(1,2.01,0.1)

for i,date in enumerate(dates_loop):
    for fac in evap_factors:
        date_debut=date
        print (date_debut,i)
        print('Evaporation factor ', fac)
        date_only = date.strftime('%Y-%m-%d')
        date_fin='2024-12-01'
        date_fin = pd.Timestamp(date_fin)
        surf_abbe = data_surf_lac_abbe.loc[date_debut:date_fin]
        surf_abbe = surf_abbe['Surf_water (km2)']
        surf_abbe.name = 'Surface Water'
        climatology = surf_abbe.groupby(surf_abbe.index.month).mean()
        surf_abbe_filled = surf_abbe.copy()
        mask = surf_abbe.isna()
        surf_abbe_filled.loc[mask] = surf_abbe.index[mask].month.map(climatology)

        surf_nlakes = S_lakeN_nan.loc[date_debut:date_fin]
        surf_nlakes = surf_nlakes['Surf_water (km2)']
        surf_nlakes.name = 'Surface Water'
        climatology = surf_nlakes.groupby(surf_nlakes.index.month).mean()
        surf_nlakes_filled = surf_nlakes.copy()
        mask = surf_nlakes.isna()
        surf_nlakes_filled.loc[mask] = surf_nlakes.index[mask].month.map(climatology)

    #filtrage temporelle des données
        E = data_evap_gleam_abbe[date_debut:date_fin]
        E=E*fac # Scalar factor
        P = data_pluie_CHIRPS_mois_abbe[date_debut:date_fin]
        Q=data_debitp90["Q_TENDAHO (m3)"][date_debut:date_fin]
        Q.index = Q.index.to_period('M').to_timestamp('D')
        S=surf_nlakes_filled + surf_abbe_filled
        S_abbe=surf_abbe_filled
        S_lakeN=surf_nlakes_filled
        h=data_h_level['h water level (m)'][date_debut:date_fin]

        df_data_compile=pd.concat([S,S_abbe,S_lakeN,Q,h,P,E],axis=1) #concaténation
        df_data_compile.columns=['Surf_water (km2)','Surf_water_abbe (km2)','Surf_water_lakeN (km2)',
                             'Q_TENDAHO (m3)','h water level (m)','flux precip lacs (mm)',
                             'flux evap lacs (mm)']
    #### ---- Calling the Lake model ---- ####

    # -- Correcting NaN values -- #

        df_data=df_data_compile.copy()
        df_data=fill_nan_with_avg(df_data,'Surf_water (km2)')
        df_data=fill_nan_with_avg(df_data,'Surf_water_lakeN (km2)')
        df_data=fill_nan_with_avg(df_data,'Surf_water_abbe (km2)')

        date_debut = pd.to_datetime(date_debut)
        closest_date = opw_area_dict['Abbe'].index.get_indexer([date_debut], method='nearest')

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

        df_model_orig,h_etablir_relation_abbe,S_etablir_relation_abbe,V_etablir_relation_abbe,h_etablir_relation_lakeN,S_etablir_relation_lakeN,V_etablir_relation_lakeN,rmse_abbe,rmse_lakeN=model(MNT_path_abbe_3,MNT_path_lakeN,df_data,"Q_TENDAHO (m3)","flux precip lacs (mm)","flux evap lacs (mm)",m_param,h_ini_abbe=h_ini_abbe,S_ini_abbe=S_ini_abbe,V_ini_abbe=V_ini_abbe,h_ini_lakeN=h_ini_lakeN,S_ini_lakeN=S_ini_lakeN,V_ini_lakeN=V_ini_lakeN,seuil_alt_niveau_eau_abbe=np.linspace(152, 252,110),seuil_alt_niveau_eau_lakeN=np.linspace(330, 350,20))

        results_abbe[(date_debut,fac)] = {
            "altitude_values": df_model_orig["model_hauteur_abbe (m)"].values,
            "volume_values": df_model_orig["model_Volume_abbe (m3)"].values,
            "surface_values": df_model_orig["model_Surface_abbe (m2)"].values,
            "surface_index": df_model_orig["model_Surface_abbe (m2)"].index.values,
            "rmse": rmse_abbe
        }

        results_nlakes[(date_debut, fac)] = {
            "altitude_values": df_model_orig["model_hauteur_lakeN (m)"].values,
            "volume_values": df_model_orig["model_Volume_lakeN (m3)"].values,
            "surface_values": df_model_orig["model_Surface_lakeN (m2)"].values,
            "surface_index": df_model_orig["model_Surface_lakeN (m2)"].index.values,
            "rmse": rmse_lakeN
        }
        print('RMSE Abbe: ', rmse_abbe, ' RMSE NLakes: ', rmse_lakeN, ' RMSE_Total: ', rmse_abbe+rmse_lakeN )

#### ---- Write the results ---- ####

#### ---- Figures ---- ####

keys = list(results_abbe.keys())
keys_list = list(results_abbe.keys())

# Create colormap
cmap = plt.cm.jet
colors = cmap(np.linspace(0, 1, len(keys)))

plt.figure(figsize=(14, 5))

for i,k in enumerate(keys):
    k_date, k_fac = k
    time = results_abbe[(k_date, k_fac)]['surface_index']
    abbe_lm = results_abbe[(k_date, k_fac)]['surface_values']
    label_name = f'Evap factor {k_fac:.2f}'
    col = colors[i]
    plt.plot(time, abbe_lm * 1e-6, color=col, marker='o', markerfacecolor='none', markeredgewidth=1, linewidth=1, alpha=0.4, label=label_name)

time = results_abbe[(k_date, 1.4000000000000004)]['surface_index']
abbe_lm = results_abbe[(k_date, 1.4000000000000004)]['surface_values']

plt.plot(time, abbe_lm * 1e-6, color='green', marker='x', markeredgewidth=1.5, linewidth=1.5,  label='Evap factor used')

time = results_abbe[(k_date, 1.7000000000000006)]['surface_index']
abbe_lm = results_abbe[(k_date, 1.7000000000000006)]['surface_values']

plt.plot(time, abbe_lm * 1e-6, color='tan', marker='x', markeredgewidth=1.5, linewidth=1.5,  label='Best Option')

plt.plot(opw_area_dict['Abbe'].index, opw_area_dict['Abbe'].values * 1e-6,  color = 'grey',marker='x',markeredgewidth=1.5, linewidth=1.5,  label='Zhao')

plt.plot(data_surf_lac_abbe.index, data_surf_lac_abbe.values, color = 'black',marker='x',markeredgewidth=1.5, linewidth=1.5,  label='GWSD')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Surface (km²)")
plt.title("Sensitivity to evaporation factor")
plt.savefig('Test.pdf')
plt.close()

keys = list(results_abbe.keys())
keys_list = list(results_abbe.keys())

# Create colormap
cmap = plt.cm.jet
colors = cmap(np.linspace(0, 1, len(keys)))

plt.figure(figsize=(14, 5))

for i,k in enumerate(keys):
    k_date, k_fac = k
    time = results_abbe[(k_date, k_fac)]['surface_index']
    abbe_lm = results_abbe[(k_date, k_fac)]['surface_values'] + results_nlakes[(k_date, k_fac)]['surface_values']
    label_name = f'Evap factor {k_fac:.2f}'
    col = colors[i]
    plt.plot(time, abbe_lm * 1e-6, color=col, marker='o', markerfacecolor='none', markeredgewidth=1, linewidth=1, alpha=0.4, label=label_name)

time = results_abbe[(k_date, 1.4000000000000004)]['surface_index']
abbe_lm = results_abbe[(k_date, 1.4000000000000004)]['surface_values'] + results_nlakes[(k_date, 1.4000000000000004)]['surface_values']

plt.plot(time, abbe_lm * 1e-6, color='magenta', marker='x', markeredgewidth=1.5, linewidth=1.5,  label='Evap factor used')

time = results_abbe[(k_date, 1.7000000000000006)]['surface_index']
abbe_lm = results_abbe[(k_date, 1.7000000000000006)]['surface_values'] + results_nlakes[(k_date, 1.7000000000000006)]['surface_values']

plt.plot(time, abbe_lm * 1e-6, color='tan', marker='x', markeredgewidth=1.5, linewidth=1.5,  label='Best Option')

plt.plot(opw_area_dict['Abbe'].index, (opw_area_dict['Abbe'].values + opw_area_dict['Gemeri'].values + opw_area_dict['Afambo'].values) * 1e-6,  color = 'grey',marker='x',markeredgewidth=1.5, linewidth=1.5,  label='Zhao')

plt.plot(data_surf_lac_abbe.index, data_surf_lac_abbe.values + data_surf_lac_Gemeri.values + data_surf_lac_Afambo, color = 'black',marker='x',markeredgewidth=1.5, linewidth=1.5,  label='GWSD')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Surface (km²)")
plt.title("Sensitivity to evaporation factor")
plt.savefig('EvapSensTest_AllLakes.pdf')
plt.close()

ds = xr.Dataset(
    {
        "abbe_lm": (["time"], abbe_lm),
        "nlakes_lm": (["time"], nlakes_lm),
    },
    coords={
        "time": time
    }
)

ds["abbe_lm"].attrs["units"] = "m2"   # adapt if needed
ds["nlakes_lm"].attrs["units"] = "count"

ds["time"].attrs["standard_name"] = "time"

ds.to_netcdf("LakeModel_CHIRPS-GLEAM.nc")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(14, 5))

# --- Data ---
data_surf_lac_abbe2 = df['abbe_landsat8-9']
data_surf_lac_Gemeri2 = df['gemeri_landsat8-9']
data_surf_lac_Afambo2 = df['afambo_landsat8-9']

time = results_abbe[(k_date, 1.4000000000000004)]['surface_index']
abbe_lm = (
    results_abbe[(k_date, 1.4000000000000004)]['surface_values']
    + results_nlakes[(k_date, 1.4000000000000004)]['surface_values']
)

# --- Plot ---
plt.plot(time, abbe_lm * 1e-6, color='red', marker='x',
         markersize=8, markerfacecolor='none', linewidth=1.5,
         markeredgewidth=1.5, label='Lake Model')

plt.plot(opw_area_dict['Abbe'].index,
         (opw_area_dict['Abbe'].values +
          opw_area_dict['Gemeri'].values +
          opw_area_dict['Afambo'].values) * 1e-6,
         color='blue', marker='o', markersize=8,
         markerfacecolor='none', markeredgewidth=1.2,
         linewidth=1.5, label='Zhao')

plt.plot(data_surf_lac_abbe.index,
         data_surf_lac_abbe.values +
         data_surf_lac_Gemeri.values +
         data_surf_lac_Afambo.values,
         color='black', marker='o', markersize=8,
         markerfacecolor='none', markeredgewidth=1.2,
         linewidth=1.5, label='GWSD Mullen Correction')

plt.plot(data_surf_lac_abbe.index,
         data_surf_lac_abbe2.values +
         data_surf_lac_Gemeri2.values +
         data_surf_lac_Afambo2.values,
         color='darkgray', marker='o', markersize=8,
         markerfacecolor='none', markeredgewidth=1.2,
         linewidth=1.5, label='Landsat8-9 Mullen Correction')

# --- Axis formatting ---
ax = plt.gca()

# Major ticks every 5 years
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Minor ticks every year
ax.xaxis.set_minor_locator(mdates.YearLocator(1))

# --- Gridlines ---
ax.grid(which='major', linestyle='-', linewidth=0.6, alpha=0.7)
ax.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.5)

# --- Labels & title (bigger fonts) ---
plt.xlabel("Time", fontsize=16)
plt.ylabel("Surface (km²)", fontsize=16)
plt.title("Afambo-Gemeri-Abhe Lake Surface", fontsize=18)

# Tick label size
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Legend
plt.legend(fontsize=13)

# Optional: rotate x labels slightly (often helps readability)
plt.xticks(rotation=45)
ax.set_xlim(pd.Timestamp("1985-01-01"), pd.Timestamp("2024-12-31"))
plt.tight_layout()
plt.savefig('SurfaceTimeSeries_AllLakes.pdf')
plt.close()

plt.figure(figsize=(14, 5))

# --- Data ---
data_surf_lac_abbe2 = df['abbe_landsat8-9']
data_surf_lac_Gemeri2 = df['gemeri_landsat8-9']
data_surf_lac_Afambo2 = df['afambo_landsat8-9']

time = results_abbe[(k_date, 1.4000000000000004)]['surface_index']
abbe_lm = (
    results_abbe[(k_date, 1.4000000000000004)]['surface_values']
)

# --- Plot ---
plt.plot(time, abbe_lm * 1e-6, color='red', marker='x',
         markersize=8, markerfacecolor='none', linewidth=1.5,
         markeredgewidth=1.5, label='Lake Model')

plt.plot(opw_area_dict['Abbe'].index,
         opw_area_dict['Abbe'].values* 1e-6,
         color='blue', marker='o', markersize=8,
         markerfacecolor='none', markeredgewidth=1.2,
         linewidth=1.5, label='Zhao')
         
plt.plot(data_surf_lac_abbe.index,
         data_surf_lac_abbe.values,
         color='black', marker='o', markersize=8,
         markerfacecolor='none', markeredgewidth=1.2,
         linewidth=1.5, label='GWSD Mullen Correction')
         
plt.plot(data_surf_lac_abbe.index,
         data_surf_lac_abbe2.values,
         color='darkgray', marker='o', markersize=8,
         markerfacecolor='none', markeredgewidth=1.2,
         linewidth=1.5, label='Landsat8-9 Mullen Correction')

# --- Axis formatting ---
ax = plt.gca()

# Major ticks every 5 years
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Minor ticks every year
ax.xaxis.set_minor_locator(mdates.YearLocator(1))

# --- Gridlines ---
ax.grid(which='major', linestyle='-', linewidth=0.6, alpha=0.7)
ax.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.5)

# --- Labels & title (bigger fonts) ---
plt.xlabel("Time", fontsize=16)
plt.ylabel("Surface (km²)", fontsize=16)
plt.title("Abhe Lake Surface", fontsize=18)

# Tick label size
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Legend
plt.legend(fontsize=13)

# Optional: rotate x labels slightly (often helps readability)
plt.xticks(rotation=45)
ax.set_xlim(pd.Timestamp("1985-01-01"), pd.Timestamp("2024-12-31"))
plt.tight_layout()
plt.savefig('SurfaceTimeSeries_Abhe.pdf')
plt.close()


