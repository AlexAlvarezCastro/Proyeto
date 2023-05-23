# %% [markdown]
# # 1. Lirerías y Datos

# %%
import pandas as pd 
import numpy as np 
import folium
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt
import fiona
from shapely.geometry import Point
from scipy.stats import pearsonr
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

# %%
crimes = pd.read_csv('Datos/Crimes_-_2001_to_Present.csv')
#Fuente: Chicago data portal --> https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2

codigos = pd.read_csv('Datos/Chicago_Police_Department_-_Illinois_Uniform_Crime_Reporting__IUCR__Codes.csv')
#Fuente: Chicago data portal --> https://data.cityofchicago.org/Public-Safety/Chicago-Police-Department-Illinois-Uniform-Crime-R/c7ck-438e

todo = pd.read_csv('Datos/todo.csv')

gdf = gpd.read_file('Datos/Boundaries - Community Areas (current).geojson')


# %% [markdown]
# # 2. EDA

# %%
print(f'El número de filas es {crimes.shape[0]} y el número de columnas {crimes.shape[1]}')

crimes.head(5)

# %%
unicos = len(crimes.ID.unique())

print(f'Hay {unicos} observaciones únicas en la columna ID, que es igual que el número de observciones ({crimes.shape[0]}), por lo tanto es una PK válida')

# %%
socioeconomic.drop(socioeconomic[socioeconomic['index'] == 'CHICAGO'].index, axis=0, inplace=True)
socioeconomic.drop('Unnamed: 0', axis=1, inplace=True)
socioeconomic.sort_values(by='Nº', inplace=True)
socioeconomic.set_index(['Nº'], inplace=True)

# %%
gdf['area_num_1'] = gdf['area_num_1'].astype(int)
gdf.sort_values(by='area_num_1', inplace=True)
gdf.set_index('area_num_1', inplace=True)

# %%
crimes.dropna(subset=['Location'], inplace=True)

crimes.Date = pd.to_datetime(crimes.Date, format='%m/%d/%Y %I:%M:%S %p')
crimes['hour'] = crimes.Date.dt.hour
crimes['day'] = crimes.Date.dt.day_of_week
crimes['month'] = crimes.Date.dt.month
crimes['year'] = crimes.Date.dt.year

inicial = crimes['Location Description'].isnull().sum()
print(f'Valores perdidos iniciales: {inicial}')
moda = crimes['Location Description'].mode().values[0]
print(f'Moda de la variable Year: {moda}')
crimes['Location Description']= crimes['Location Description'].replace(np.nan, moda)
final = crimes['Location Description'].isnull().sum()
print(f'Valores perdidos finales: {final}')

crimes.isnull().sum()

# %%
print(f'El número de filas es {crimes.shape[0]} y el número de columnas {crimes.shape[1]}')

# %%
# filtro = '2016-01-01 00:00:00' > crimes['Date'] > '2020-01-01 00:00:00'

# crimes = crimes.loc[(crimes['Date'] >= '2016-01-01 00:00:00') & (crimes['Date'] <= '2020-01-01 00:00:00')]


# %% [markdown]
# # 3. Mapa de calor

# %% [markdown]
# Barrios más conlictivos según FBI
# 
# 1. Parque West Garfield (Distrito 26)
# 2. East Garfield Park (Distrito 27)
# 3. Washington Park (Distrito 6) 
# 
# 

# %%
heatmap_data = crimes.groupby("Community Area").size().reset_index(name="count")
heatmap_data.set_index('Community Area', inplace=True)
heatmap_data.drop(heatmap_data[heatmap_data.index == 0].index, axis=0, inplace=True)
heatmap_data

fig1 = px.choropleth_mapbox(heatmap_data, geojson=gdf, locations= heatmap_data.index, color='count',
                           #color_continuous_scale="Viridis", #range_color=(0, 100),
                           mapbox_style="carto-positron",
                           zoom=8.4, center={"lat": 41.881832, "lon": -87.623177},
                           opacity=0.5,
                           labels={'Community Area':'Distrito', 'count':'Número de observaciones'})
fig1 = fig1.update_layout(title={'text':'Crímenes por Área Comunitaria', 'font': {'size': 24}}, title_x=0.5, title_y=0.95)
fig1.show()


# %%
# rango_de_años = np.arange(crimes['Date'].min().year, crimes['Date'].max().year + 1, 7)

# # Crea un bucle for que itere sobre los rangos de años
# for i in range(len(rango_de_años) - 1):
#     # Filtra el DataFrame para obtener las observaciones correspondientes al rango de años actual
#     crimes_actual = crimes[(crimes['Date'].dt.year >= rango_de_años[i]) & (crimes['Date'].dt.year < rango_de_años[i+1])]
    
#     # Crea la visualización utilizando Plotly
#     heatmap_data = crimes_actual.groupby("Community Area").size().reset_index(name="count")

#     fig = px.choropleth_mapbox(heatmap_data, geojson=gdf, locations=gdf.index, color="count",
#                            color_continuous_scale="Viridis", #range_color=(0, 100),
#                            mapbox_style="carto-positron",
#                            zoom=8.4, center={"lat": 41.881832, "lon": -87.623177},
#                            opacity=0.5,
#                            labels={'Community Area':'Distrito', 'count':'Número de observaciones'})
#     fig.update_layout(title='Visualización para el rango de años ' + str(rango_de_años[i]) + '-' + str(rango_de_años[i+1]))
#     fig.show()


# %% [markdown]
# # 4. Evolución temporal/tipos de delitos

# %%
fig = px.histogram(crimes, x='year', nbins=23, title='Distribución de observaciones por año',
                   labels={'year': 'Año', 'count': 'Número de observaciones'})
#fig.add_trace(px.line(crimes.groupby('year').size().reset_index(name='count'), x='year', y='count').data[0])
fig.show()

# %% [markdown]
# # 5. Correlaciones

# %%
todo

# %%
subset1 = heatmap_data[['count']]
subset2 = socioeconomic[['PERCENT OF HOUSING CROWDED', 'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 16+ UNEMPLOYED', 
                         'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA', 'PERCENT AGED UNDER 18 OR OVER 64', 'PER CAPITA INCOME ', 'HARDSHIP INDEX']]
subset2 = pd.concat([todo, subset2], axis=1)

subset = pd.concat([subset1, subset2], axis=1)

subset.drop('Unnamed: 0', axis=1, inplace=True)

# %%
corr_matrix = subset.corr()
corr_matrix

# %%
fig = go.Figure(
    go.Heatmap(
        z=corr_matrix.values,  # Los datos son la matriz de correlación
        x=corr_matrix.columns, # Las etiquetas del eje x son las columnas
        y=corr_matrix.index,   # Las etiquetas del eje y son los índices
        colorscale='RdBu',     # Elige un mapa de color
        colorbar=dict(title='Correlación'),  # Añade una etiqueta a la barra de color
    )
)

# Personalizar la figura
fig.update_layout(
    width=800,
    height=800,
    title='Matriz de correlación',
    xaxis=dict(title='Variables'),
    yaxis=dict(title='Variables'),
)

# %%
# for i in subset.columns:
#     for j in subset.columns[1:]:
#         corr, p_value = pearsonr(subset[i] , subset[j])
#         #print(f"Coeficiente de correlación de Pearson para variables {i} y {j}: {corr}")
#         if p_value < 0.05:
#             print(f"Valor p para variables {i} y {j}: {p_value}")
#         else: 
#             print("No relación")

# for i in subset.columns:
#     corr, p_value = pearsonr(subset['count'] , subset[i])
#     print(f"Coeficiente de correlación de Pearson para variables count y {i}: {corr}")
#     print(f"Valor p para variables count y {i}: {p_value}")


# %%
correlation_matrix, pvalue_matrix = spearmanr(subset)
df = pd.DataFrame(correlation_matrix, columns=subset.columns, index=subset.columns)

# %%
correlation_matrix

# %%
fig2 = go.Figure(
    go.Heatmap(
        z=df.values,  # Los datos son la matriz de correlación
        x=df.columns, # Las etiquetas del eje x son las columnas
        y=df.index,   # Las etiquetas del eje y son los índices
        colorscale='RdBu',     # Elige un mapa de color
        colorbar=dict(title='Correlación'),  # Añade una etiqueta a la barra de color
    )
)

# Personalizar la figura
fig2.update_layout(
    width=800,
    height=800,
    title='Matriz de correlación',
    xaxis=dict(title='Variables'),
    yaxis=dict(title='Variables'),
)


