import pandas as pd
import plotly.graph_objects as go
import numpy as np

import dash
from dash import html, dcc


from dash.dependencies import Input, Output


# Load and prepare datasets
df = pd.read_csv(r'Datasets\electricity_data.csv')
df = df[1110:]
df.reset_index(drop=True, inplace=True)
df.index = pd.to_datetime(df['Date'])

df2 = pd.read_csv(r'Datasets\electrical_appliance_consumption.csv')
df2 = df2[df2['year'] == 2021]
df2.reset_index(drop=True, inplace=True)

df3 = pd.read_csv(r'Datasets\electrical_forecast.csv')

df4 = pd.read_csv(r'Datasets\electricity_appliance_wise_data.csv')
df4['Date'] = pd.to_datetime(df4['Date'])
df4 = df4[df4['Date'].dt.year == 2021]
df4.reset_index(drop=True, inplace=True)

# Initialize Dash app
app = dash.Dash(__name__)
app.title = 'Electra.AI'

party_data = [{'label': x, 'value': x} for x in [
    'Time-Series Plot', 'Appliance-wise Consumption',
    'Electricity Consumption Forecast', 'Faulty Devices']]

# Define layout
app.layout = html.Div(children=[
    html.Div(className='row', children=[
        html.Div(className='four columns div-user-controls', children=[
            html.H2('Electra.AI Dashboard', style={'font-family': 'Trebuchet MS'}),
            html.Div(className='div-for-dropdown', children=[
                dcc.Dropdown(id='stockselector',
                             options=party_data,
                             value='Time-Series Plot',
                             style={'backgroundColor': '#1E1E1E'},
                             placeholder="Select an Option")
            ], style={'color': '#1E1E1E'})
        ]),
        html.Div(className='eight columns div-for-charts bg-grey', children=[
            dcc.Graph(id='timeseries', config={'displayModeBar': False})
        ]),
    ])
])

# Define callback to update graph
@app.callback(Output('timeseries', 'figure'),
              [Input('stockselector', 'value')])
def update_timeseries(selected_dropdown_value):
    if selected_dropdown_value == 'Electricity Consumption Forecast':
        df_sub = df3
        df_anoms = df_sub[df_sub['MAE'] >= 15]
        df_anoms.reset_index(drop=True, inplace=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Total_Consumption'],
                                 mode='lines', name='Actual Consumption',
                                 line_color="#19E2C5"))
        fig.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Predicted_Consumption'],
                                 mode='lines', name='Predicted Consumption',
                                 line_color="#C6810B"))
        fig.add_trace(go.Scatter(x=df_anoms['Date'], y=df_anoms['Total_Consumption'],
                                 mode='markers', name='Excess Consumption'))
        fig.update_traces(marker=dict(size=5,
                                      line=dict(width=5, color='#C60B0B')))
        fig.update_layout(template='plotly_dark',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          margin={'b': 15},
                          autosize=True,
                          yaxis_title="Consumption (kWh)",
                          xaxis_title="Date",
                          title={'text': 'Time-Series Plot & Forecasting Electricity Consumption for this year',
                                 'font': {'color': 'white'}, 'x': 0.5})
        return fig

    elif selected_dropdown_value == 'Faulty Devices':
        df_sub = df4

        def zscore(x, window):
            r = x.rolling(window=window)
            m = r.mean().shift(1)
            s = r.std(ddof=0).shift(1)
            z = (x - m) / s
            return z

        df_sub['kap_zscore'] = zscore(df_sub['Kitchen Appliances'], 30)
        df_sub['fridge_zscore'] = zscore(df_sub['Fridge'], 30)
        df_sub['ac_zscore'] = zscore(df_sub['AC'], 30)
        df_sub['oap_zscore'] = zscore(df_sub['Other Appliances'], 30)
        df_sub['wm_zscore'] = zscore(df_sub['Washing Machine'], 3)

        df_sub.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_sub.fillna(0, inplace=True)

        df_anom_kap = df_sub[df_sub['kap_zscore'] > 5]
        df_anom_fridge = df_sub[df_sub['fridge_zscore'] > 5]
        df_anom_ac = df_sub[df_sub['ac_zscore'] > 5]
        df_anom_oap = df_sub[df_sub['oap_zscore'] > 5]
        df_anom_wm = df_sub[df_sub['wm_zscore'] > 5]

        fig_anom = go.Figure(data=[
            go.Scatter(x=df_sub['Date'], y=df_sub['Fridge'],
                       mode='lines', name='Fridge Consumption', line_color="#19E2C5"),
            go.Scatter(x=df_anom_fridge['Date'], y=df_anom_fridge['Fridge'],
                       mode='markers', name='Fridge Fluctuations', marker=dict(size=10)),
            go.Scatter(x=df_sub['Date'], y=df_sub['AC'],
                       mode='lines', name='AC Consumption', line_color="#636EFA"),
            go.Scatter(x=df_anom_ac['Date'], y=df_anom_ac['AC'],
                       mode='markers', name='AC Fluctuations', marker=dict(size=10)),
            go.Scatter(x=df_sub['Date'], y=df_sub['Other Appliances'],
                       mode='lines', name='Other Appliances', line_color="#EF553B"),
            go.Scatter(x=df_anom_oap['Date'], y=df_anom_oap['Other Appliances'],
                       mode='markers', name='Other Appliances Fluctuations', marker=dict(size=10)),
            go.Scatter(x=df_sub['Date'], y=df_sub['Kitchen Appliances'],
                       mode='lines', name='Kitchen Appliances', line_color="#00CC96"),
            go.Scatter(x=df_anom_kap['Date'], y=df_anom_kap['Kitchen Appliances'],
                       mode='markers', name='Kitchen Appliances Fluctuations', marker=dict(size=10)),
            go.Scatter(x=df_sub['Date'], y=df_sub['Washing Machine'],
                       mode='lines', name='Washing Machine', line_color="#AB63FA"),
            go.Scatter(x=df_anom_wm['Date'], y=df_anom_wm['Washing Machine'],
                       mode='markers', name='Washing Machine Fluctuations', marker=dict(size=10))
        ])

        fig_anom.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=list([
                        dict(label="Fridge",
                             method="update",
                             args=[{"visible": [True, True] + [False]*8},
                                   {"title": "Anomalies in power consumption of Fridge"}]),
                        dict(label="AC",
                             method="update",
                             args=[{"visible": [False, False, True, True] + [False]*6},
                                   {"title": "Anomalies in power consumption of AC"}]),
                        dict(label="Other Appliances",
                             method="update",
                             args=[{"visible": [False]*4 + [True, True] + [False]*4},
                                   {"title": "Anomalies in power consumption of Other Appliances"}]),
                        dict(label="Kitchen Appliances",
                             method="update",
                             args=[{"visible": [False]*6 + [True, True] + [False]*2},
                                   {"title": "Anomalies in power consumption of Kitchen Appliances"}]),
                        dict(label="Washing Machine",
                             method="update",
                             args=[{"visible": [False]*8 + [True, True]},
                                   {"title": "Anomalies in power consumption of Washing Machine"}]),
                    ]),
                )
            ],
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            title_font_color="#90E219",
            font_color="#90E219",
            autosize=True
        )
        return fig_anom

    elif selected_dropdown_value == 'Appliance-wise Consumption':
        df_sub = df2

        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
        appliances = ['Fridge', 'Kitchen Appliances', 'AC', 'Washing Machine', 'Other Appliances']

        irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                         'rgb(175, 49, 35)', 'rgb(36, 73, 147)']

        pies = []
        for month in months:
            values = [df_sub[df_sub['month'] == month][app].values[0] for app in appliances]
            pies.append(go.Pie(
                name=f'{month} 2021',
                labels=appliances,
                values=values,
                marker_colors=irises_colors,
                hole=0.3,
                domain={'x': [months.index(month) / len(months), (months.index(month) + 1) / len(months)]}
            ))

        layout = go.Layout(
            title={'text': 'Electricity Consumption of Appliances in House (Monthly in 2021)',
                   'font': {'color': 'white'}, 'x': 0.5},
            annotations=[{
                'font': {'size': 20},
                'showarrow': False,
                'text': month,
                'x': (months.index(month) + 0.5) / len(months),
                'y': 0.5
            } for month in months],
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font_color='white',
            autosize=True
        )

        fig = go.Figure(data=pies, layout=layout)
        return fig

    else:  # Default: Time-Series Plot
        df_sub = df
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sub.index, y=df_sub['Total_Consumption'],
                                 mode='lines', name='Electricity Consumption',
                                 line_color="#19E2C5"))

        fig.update_layout(template='plotly_dark',
                          paper_bgcolor='rgba(0, 0, 0, 0)',
                          plot_bgcolor='rgba(0, 0, 0, 0)',
                          margin={'b': 15},
                          autosize=True,
                          yaxis_title="Consumption (kWh)",
                          xaxis_title="Date",
                          title={'text': 'Time-Series Plot of Electricity Consumption',
                                 'font': {'color': 'white'}, 'x': 0.5})
        return fig


if __name__ == '__main__':
    app.run(debug=True)
