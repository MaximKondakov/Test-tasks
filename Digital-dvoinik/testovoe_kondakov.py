import dash
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from geopy import geocoders 
from geopy.geocoders import Nominatim

df = pd.read_csv(
    r'C:\Users\Максим\Desktop\forFBpost.csv', delimiter=';' 
)

geolocator = Nominatim(user_agent="Tester")
res={}
city_list = df['Город'].unique()

for city in city_list:
  location = geolocator.geocode(city)
  res[city] = (location.latitude, location.longitude)

df['longitude'] = df.apply(lambda x: res.get(x['Город'])[1], axis=1)
df['latitude'] = df.apply(lambda x: res.get(x['Город'])[0], axis=1)

def another_graph():
    data1 = df.copy()
    data1['grow_model'] = data1.groupby('Город')['Модель'].transform(lambda x: 0 if (x.iloc[-1] - x.iloc[0])<0 else 1)
    y1 = data1['grow_model'].mean()

    data2 = df.loc[~df['fact'].isna()].reset_index(drop=True)
    data2['grow'] = data2.groupby('Город')['fact'].transform(lambda x: 0 if (x.iloc[-1] - x.iloc[0])<0 else 1)
    y2 = data2['grow'].mean()

    fig = go.Figure()
    fig.add_bar(x = [0], y = [y1], name = 'процент модели')
    fig.add_bar(x = [1], y = [y2], name = 'процент факт')
    fig.update_layout(
            {
                "title": "процент растущих городов по факту и по результатам модели",
                "yaxis": {"title": "процент растущих городов"},   
            })    
    return fig    


app = Dash(__name__)

app.layout = html.Div(
    [
    html.H1(id = 'H1', children = 'Testovoe zadanie', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),

    html.H4("Отображение городов, которые выросли\вымирают по фактической численности"),
    dcc.RadioItems(
        id="radio_map",
        inline=True,
        options=["all", "grow", "negative_grow"],
        value="all",
    ),       
    dcc.Graph(id = 'map_plot'),
    dcc.Markdown(
                "Вымирающие города не зависят от местоположения"
                    ),

    html.H4("Выбор города"),
    dcc.Dropdown( id = 'dropdown',
    options = df['Город'].unique(),
    value = df['Город'].unique()[0]),

    dcc.Markdown(id = 'show_people'),

    html.H4("График фактической численности выбранного города, с прогнозом модели"), 
    dcc.Graph(id = 'bar_plot'),

    dcc.Markdown(
                """
                Если модель - регрессия, то она переобучилась на выборке 2008-2020, поэтому слишком уверена в прогнозах.\n
                Для прогноза численности города нужно учитывать многие факторы, такие факторы как коэффициент рождаемости\n
                и миграцию населения.\n
                Так же для предсказания на длительном промежутке времени должна возрастать дисперсия прогноза, чего нет в данной модели\n\n
                Так как в предложенном датасете мало данных, для прогноза численности города я бы использовал 2 модели \n
                1) дискретная модель на прогноз численности населения города, используя коэффициент рождаемости/смертности\n
                2) модель миграции, которая зависит от расстояния до ближайшего миллионника, и средней зп по региону
                """
                    ),
    html.H4("Распределение городов по численности, которые выросли"),
    dcc.Slider(2, 15, 1, value=7, id="n_bins"),
    dcc.Graph(
        id='hist_id'
    ),  
    dcc.Markdown(
                """
                на этом графике видна прямая зависимость - чем больше численность города, тем больше городов показали рост населения,\n
                в то время как маленькие города вымирают
                """
                    ),
    dcc.Graph(id ='qwe',figure=another_graph()
    ),   
    html.H4("Таблица выбранного города"),
    dash_table.DataTable(
            id='my_datatable',
            columns=[{'name': i, 'id': i} for i in df.columns]
        )
    ])


@app.callback(
    Output("hist_id", "figure"),
    Input("n_bins", "value"),)
def callback_hist(n_bins):
    data = df.loc[~df['fact'].isna()].reset_index(drop=True)

    data['grow'] = data.groupby('Город')['fact'].transform(
        lambda x: 0 if (x.iloc[-1] - x.iloc[0])<0 else 1)    

    data['fact_bin'] = pd.qcut(data['fact'], n_bins, duplicates='drop')
    plot_df = data.groupby('fact_bin')['grow'].agg(['mean'])

    fig = px.bar(
        plot_df,
        x=plot_df.index.astype(str),
        y='mean',
        orientation="v") 
    fig.update_xaxes(type='category')
    fig.update_layout(
        {
            "title": f"number of bins {n_bins}",
            "yaxis": {"title": "процент городов, у которых рост населения"},
            "xaxis": {"title": "численность людей в интервале"},      
        })
    return fig


@app.callback(
    Output("show_people", "children"),
    Input("dropdown", "value"),)
def update_output(dropdown_value):
    x = df.loc[(df['Город'] == dropdown_value) & (df['year'] == 2020)]
    xx = df.loc[(df['Город'] == dropdown_value) & (~df['fact'].isna())]
    xx['diff_fact_year']  = xx['fact'].diff().fillna(0)          
    text = f"""
    В городе {dropdown_value} на 2020 год проживает {x.fact.values} человек. \n
    Изменения с 2008 по 2020 составило = {xx.diff_fact_year.sum()}
    """
    return text


@app.callback(
    Output("map_plot", "figure"),
    Input("dropdown", "value"),
    Input("radio_map", "value"),    
    )
def generate_chart(values, radio_value):
    chosen_city = df.loc[df['Город'] == values].head(1) 
    df_map = df.loc[~df['fact'].isna()].reset_index(drop=True)

    df_map['grow'] = df_map.groupby('Город')['fact'].transform(
        lambda x: 0 if (x.iloc[-1] - x.iloc[0])<0 else 1)

    if radio_value == "all":                   
        fig = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            hover_data=["Город", "fact"],
            color="fact",
            size="fact",
            zoom=8,
            size_max = 50,
            center= {
                'lat': chosen_city['latitude'].values[0],
                'lon': chosen_city['longitude'].values[0]} 
        ) 
    elif radio_value == "grow":
        df_map = df_map.loc[df_map['grow']==1]        
        fig = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            hover_data=["Город", "fact"],
            color="fact",
            size="fact",
            zoom=8,
            size_max = 50,
            center= {
                'lat': chosen_city['latitude'].values[0],
                'lon': chosen_city['longitude'].values[0]} 
        ) 
    elif radio_value == "negative_grow":
        df_map = df_map.loc[df_map['grow']==0]   
        fig = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            hover_data=["Город", "fact"],
            color="fact",
            size="fact",
            zoom=8,
            size_max = 30,
            center= {
                'lat': chosen_city['latitude'].values[0],
                'lon': chosen_city['longitude'].values[0]} 
        ) 


    fig.update_layout(mapbox_style="open-street-map")
    return fig 

@app.callback(Output('my_datatable', 'data'),
              [Input('dropdown', 'value')])
def update_rows(selected_value):
    data = df.loc[df['Город']=='{}'.format(selected_value)].to_dict('records')
    return data    
    
@app.callback(Output(component_id='bar_plot', component_property= 'figure'),
              [Input(component_id='dropdown', component_property= 'value')])
def graph_update(dropdown_value):
    print(dropdown_value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df.loc[df['Город']=='{}'.format(dropdown_value)]['year'],       
                y = df.loc[df['Город']=='{}'.format(dropdown_value)]['Модель'],
                line = dict(color = 'firebrick', width = 2),
                mode='lines',
                name='Модель'
                )) 

    fig.add_trace(go.Scatter(x = df.loc[df['Город']=='{}'.format(dropdown_value)]['year'],       
                y = df.loc[df['Город']=='{}'.format(dropdown_value)]['fact'],
                line = dict(color = 'blue', width = 4),
                mode='lines+markers',
                name='Факт'                
                ))   

    fig.add_trace(go.Scatter(x = df.loc[df['Город']=='{}'.format(dropdown_value)]['year'],       
                y = df.loc[df['Город']=='{}'.format(dropdown_value)]['Нижняя граница'],
                line = dict(color = 'indigo', width = 1),
                mode='lines',
                name='нижняя'                  
                ))     

    fig.add_trace(go.Scatter(x = df.loc[df['Город']=='{}'.format(dropdown_value)]['year'],       
            y = df.loc[df['Город']=='{}'.format(dropdown_value)]['Верхняя граница'],
            line = dict(color = 'indigo', width = 1),
            mode='lines',
            name='верхняя',
            fill='tonexty'                    
            ))                                                       
  
    fig.update_layout(title = 'Population over time',
                      xaxis_title = 'year',
                      yaxis_title = 'Population'
                      )
    return fig  


if __name__ == "__main__":
    app.run_server(debug=True)

    