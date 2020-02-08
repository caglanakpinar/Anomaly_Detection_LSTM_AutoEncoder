import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import random
import plotly.graph_objs as go
import plotly.offline as offline
import datetime
import webbrowser

from configs import sample_sizes, k_means_cluster_colors, related_cols
from configs import feature, models_output, related_columns, is_local_run, features_cols_2


def get_last_day_comparisions(data):
    data['amount_mean'], data['amount_max'], data['amount_min'] = data['Amount'], data['Amount'], data['Amount']
    data['amount_total'] = data['Amount']
    data['amount_median'] = data['Amount']
    data['transaction_count'] = data['PaymentTransactionId']
    data['day'] = data['Created_Time'].apply(lambda x: datetime.datetime.strptime(str(x)[0:10], '%Y-%m-%d'))
    data_pv = data.pivot_table(index=['customer_id', 'day'], aggfunc={'amount_mean': 'mean',
                                                                      'amount_median': 'median',
                                                                      'amount_max': 'max',
                                                                      'amount_min': 'min',
                                                                      'amount_total': 'sum',
                                                                      'transaction_count': 'count',
                                                                      }).reset_index()
    return data_pv.sort_values(by=['customer_id', 'day'])


def calculate_last_day_differences(df, calculate_cols):
    # Prev Daily Transaction To Last Day Transaction
    df[calculate_cols + '_prev'] = df.sort_values(['day', 'customer_id'], ascending=True).groupby('customer_id')[
        calculate_cols].shift(1)
    treatment_2 = df[df['is_last_day'] == 1]
    treatment_2[calculate_cols + '_last_2_days_diff'] = treatment_2.apply(lambda row:
                                                                          row[calculate_cols] - row[
                                                                              calculate_cols + '_prev'] if row[
                                                                                                               calculate_cols + '_prev'] ==
                                                                                                           row[
                                                                                                               calculate_cols + '_prev'] else None,
                                                                          axis=1)
    df = pd.merge(df, treatment_2[['customer_id', calculate_cols + '_last_2_days_diff']], on='customer_id', how='left')

    # Last Day Transaction to Historic Daily Max Transaction Count to Last Day Of Trasaction Count
    treatment_4 = df.query("is_last_day == 0")
    treatment_4 = treatment_4.pivot_table(index=['customer_id'], aggfunc={calculate_cols: 'max'}
                                          ).reset_index().rename(columns={calculate_cols: 'max_' + calculate_cols})
    treatment_2 = pd.merge(treatment_2, treatment_4, on='customer_id', how='left')
    treatment_2[calculate_cols + '_max_to_last_day_diff'] = treatment_2.apply(
        lambda row: row['transaction_count'] - row['max_' + calculate_cols], axis=1)
    return pd.merge(df, treatment_2[['customer_id', calculate_cols + '_max_to_last_day_diff']], on='customer_id',
                    how='left')


def get_sample_from_data(main_data, sample_ratio, bias_data_condition=None):
    main_data = main_data.reset_index(drop=True)
    bias_index = list(main_data.query(bias_data_condition).reset_index()['index']) if bias_data_condition is not None else []
    sample_size = int(len(main_data) * sample_ratio)
    random_index = set(random.sample(list(range(len(main_data))), sample_size) + bias_index)
    indexes = list(main_data.index)
    print(len(list(set(random_index) & set(main_data.index))))
    return main_data.ix[list(set(random_index) & set(main_data.index))].reset_index()


def get_samples(df):
    samples_dict = {}
    for s in sample_sizes:
        samples_dict[s[0]] = get_sample_from_data(df, s[1])
    return samples_dict


def dashboard_init():
    offline.init_notebook_mode()
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    return dash.Dash(__name__, external_stylesheets=external_stylesheets)


def create_dahboard(df_train, df, features):
    features = list(feature.keys())
    df_train['label_iso'] = 0
    df_train['label_a_c'] = 0
    df_train['intersection_of_models'] = 0
    df['intersection_of_models'] = df.apply(lambda row: 1 if row['label_a_c'] == 1 and row['label_iso'] == 1 else 0, axis=1)
    df_train = pd.concat([df_train[related_cols].reset_index(drop=True),
                          df[related_cols].reset_index(drop=True)]).query("Amount == Amount")
    customer_merchant_ratios = df_train.pivot_table(index=['customer_id', 'merchant_id'],
                                                    aggfunc={'c_m_label_t_count': 'max', 'c_m_t_count': 'max',
                                                             'label_iso': 'sum'}).reset_index()
    customer_merchant_ratios['merchant_id'] = customer_merchant_ratios['merchant_id'].apply(lambda x: str(x))
    samples_dict = {}
    for s in sample_sizes:
        samples_dict[s[0]] = {'data': get_sample_from_data(df, s[1])}
        _customers = list(samples_dict[s[0]]['data']['customer_id'].unique())
        samples_dict[s[0]]['c_m_ratios'] = customer_merchant_ratios.query("customer_id in @_customers")
        samples_dict[s[0]]['customer_transactions'] = df_train.query("customer_id in @_customers")
    top_100_transactions_anomaly = df.query("intersection_of_models == 1").sort_values(by='decision_scores',
                                                                                       ascending=True).head(100)
    samples_dict['top_100'] = {'data': top_100_transactions_anomaly}
    _customers = list(samples_dict['top_100']['data']['customer_id'].unique())
    samples_dict['top_100']['c_m_ratios'] = customer_merchant_ratios.query("customer_id in @_customers")
    samples_dict['top_100']['customer_transactions'] = df_train.query("customer_id in @_customers")
    merchants = list(samples_dict['top_100']['data']['merchant_id'].unique())
    print(features)
    asd = feature
    app = dashboard_init()
    app.layout = html.Div([
        html.Div([html.H1("Anomaly Detection Multivariate Isolation Foreset - AutoEncoder")],
                 style={'textAlign': "left", "padding-bottom": "10", "padding-top": "10"}),
        html.Div(
            [html.Div(dcc.Dropdown(id="select-xaxis",
                                   options=[{'label': feature[i]['name'], 'value': i} for i in features],
                                   value=features[0], ), className="four columns",
                      style={"display": "block", "margin-left": "!%",
                             "margin-right": "auto", "width": "33%"}),
             html.Div(dcc.Dropdown(id="select-yaxis",
                                   options=[{'label': feature[i]['name'], 'value': i} for i in features],
                                   value=features[1], ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "auto", "width": "33%"}),
             html.Div(dcc.Dropdown(id="select-zaxis",
                                   options=[{'label': feature[i]['name'], 'value': i} for i in features],
                                   value=features[2], ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "1%", "width": "33%"})
             ], className="row", style={"padding": 14, "display": "block", "margin-left": "1%",
                                        "margin-right": "auto", "width": "99%"}),
        html.Div(
            [html.Div(dcc.Dropdown(id="model-selection", options=[{'label': models_output[i], 'value': i} for i in
                                                                  ['label', 'label_iso',
                                                                   'intersection_of_models']],
                                   value='label_iso', ), className="four columns",
                      style={"display": "block", "margin-left": "1%",
                             "margin-right": "auto", "width": "49%"}),
             html.Div(dcc.Dropdown(id="sample-ratio", options=[{'label': i[0].title(), 'value': i[0]} for i in
                                                               sample_sizes + [['top_100']]],
                                   value='%20', ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "auto", "width": "49%"})

             ], className="row", style={"padding": 14, "display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "99%"}),

        html.Div(
            [html.Div(dcc.Dropdown(id="customer-ids", options=[{'label': i, 'value': i} for i in ['ALL'] + _customers],
                                   value='ALL', ), className="four columns",
                      style={"display": "block", "margin-left": "1%",
                             "margin-right": "auto", "width": "49%"}),
             html.Div(dcc.Dropdown(id="merchant-ids", options=[{'label': i, 'value': i} for i in ['ALL'] + merchants],
                                   value='ALL', ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "auto", "width": "49%"})

             ], className="row", style={"padding": 14, "display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "99%"}),

        # Graphs
        html.Div([html.Div(
            [dcc.Graph(id="3d-scatter-plot", hoverData={'points': [{'PaymentTransactionId': '1213113'}]})],
            className="row", style={"padding": 0, "display": "inline-block", "width": "45%"}),
                  html.Div([dcc.Graph(id="feature-customer-merchant-ratios"),
                            dcc.Graph(id="feature-customer-transactions-amount")],
                           className="row", style={"padding": 50, "display": "inline-block", "width": "45%"}
                           )
                  ], className="row", style={"padding": 0, "display": "block", "margin-left": "auto",
                                             "margin-right": "auto", "width": "100%"}),
        # Transactions
        html.Div([html.H1("Transction Details")],
                 style={'textAlign': "left"  # , "padding-bottom": "2", "padding-top": "2"
                        }
                 ),
        html.Div([dcc.Graph(id="my-graph_2")],
                 style={"padding": 0, "display": "inline-block", "margin-left": "auto", "margin-right": "auto",
                        "width": "100%"}),
        html.Div([html.Div([dcc.Graph(id="my-graph_3")], className="row",
                           style={"padding": 0, "display": "inline-block", "width": "45%"}),
                  html.Div([dcc.Graph(id="my-graph_4")], className="row",
                           style={"padding": 50, "display": "inline-block", "width": "45%"})
                  ], className="row", style={"padding": 0, "display": "block", "margin-left": "auto",
                                             "margin-right": "auto", "width": "100%"})
    ], style={"margin-left": "1%"})

    # 3d scatter
    @app.callback(
        dash.dependencies.Output("3d-scatter-plot", "figure"),
        [dash.dependencies.Input("select-xaxis", "value"),
         dash.dependencies.Input("select-yaxis", "value"),
         dash.dependencies.Input("select-zaxis", "value"),
         dash.dependencies.Input("model-selection", "value"),
         dash.dependencies.Input("sample-ratio", "value"),
         dash.dependencies.Input("customer-ids", "value"),
         dash.dependencies.Input("merchant-ids", "value")]

    )
    def ugdate_figure(selected_x, selected_y, selected_z, model_selection, sample_sizes, customer, merchant):
        dff = samples_dict[sample_sizes]['data'].query("customer_id == @customer") if customer != 'ALL' else \
        samples_dict[sample_sizes]['data']
        dff = dff.query("merchant_id == @merchant") if merchant != 'ALL' else dff
        print(dff.head())
        dff = dff.rename(columns={selected_x: features_cols_2[selected_x], selected_y: features_cols_2[selected_y],
                                  selected_z: features_cols_2[selected_z]})

        z = dff[model_selection]
        trace = [go.Scatter3d(
                              x=dff[features_cols_2[selected_x]], y=dff[features_cols_2[selected_y]], z=dff[features_cols_2[selected_z]],
                              customdata=dff['PaymentTransactionId'],
                              mode='markers',
                              marker={'size': 3, 'color': z,
                                      'colorscale': k_means_cluster_colors,
                                      'opacity': 0.8, "showscale": False,
                                      "colorbar": {"thickness": 10, "len": 0.5, "x": 0.8, "y": 0.6, }, })]

        return {"data": trace,
                "layout": go.Layout(
                    height=700,
                    hovermode='closest',
                    scene={"aspectmode": "cube", "xaxis": {"title": f"{features_cols_2[selected_x].title()}", },
                           "yaxis": {"title": f"{features_cols_2[selected_y].title()}", },
                           "zaxis": {"title": f"{features_cols_2[selected_z].title()}", }})
                }

    # Bar Chart
    @app.callback(
        dash.dependencies.Output("feature-customer-merchant-ratios", "figure"),
        [dash.dependencies.Input("3d-scatter-plot", "hoverData"),
         dash.dependencies.Input("sample-ratio", "value")]

    )
    def get_c_m_ratios(hover_data_from_3d_scatter, sample_sizes):
        try:
            hoverData = hover_data_from_3d_scatter['points'][0]['PaymentTransactionId']
        except:
            hoverData = hover_data_from_3d_scatter['points'][0]['customdata']
        customer_id = list(samples_dict[sample_sizes]['data'][['PaymentTransactionId', 'customer_id']].query(
            "PaymentTransactionId == @hoverData")['customer_id'])[0]
        dff = samples_dict[sample_sizes]['c_m_ratios'].query("customer_id == @customer_id")
        return {"data": [
            go.Bar(name='C. Total Transaction On Merchant Label', x=dff['merchant_id'], y=dff['c_m_label_t_count']),
            go.Bar(name='C. Total Transaction On Merchant', x=dff['merchant_id'], y=dff['c_m_t_count']),
            go.Bar(name='total Anomaly', x=dff['merchant_id'], y=dff['label_iso'], marker_color='red')],
                "layout": go.Layout(height=300,
                                    title=features_cols_2['c_m_ratios'] + " || Card: " + customer_id,
                                    )
                }

    # line Chart
    @app.callback(
        dash.dependencies.Output("feature-customer-transactions-amount", "figure"),
        [dash.dependencies.Input("3d-scatter-plot", "hoverData"), dash.dependencies.Input("sample-ratio", "value")]

    )
    def get_c_m_med_amount_change_min_max_p_value(hover_data_from_3d_scatter, sample_sizes):
        try:
            hoverData = hover_data_from_3d_scatter['points'][0]['PaymentTransactionId']
        except:
            hoverData = hover_data_from_3d_scatter['points'][0]['customdata']
        print("TRANSACTION :::", hoverData)
        customer_merchant_id = list(
            samples_dict[sample_sizes]['data'][['PaymentTransactionId', 'customer_merchant_id']].query(
                "PaymentTransactionId == @hoverData")['customer_merchant_id'])[0]

        dff = \
        samples_dict[sample_sizes]['customer_transactions'].query("customer_merchant_id == @customer_merchant_id")[
            ['Amount', 'RequestInsertTime', 'label_iso']]
        # dff = pd.DataFrame(zip(customers_of_amounts[customer_merchant_id]['Amount'], customers_of_amounts[customer_merchant_id]['RequestInsertTime']))
        print(customer_merchant_id.split("_")[0])
        return {"data": [go.Scatter(x=dff['RequestInsertTime'], y=dff['Amount'], mode='markers')],
                "layout": go.Layout(height=300,
                                    title=features_cols_2[
                                              'c_m_med_amount_change_min_max_p_value'] + " || C:" +
                                        customer_merchant_id.split("_")[0] + " - M :" +
                                          customer_merchant_id.split("_")[1]
                                    )
                }

    # table
    @app.callback(
        dash.dependencies.Output("my-graph_2", "figure"),
        [dash.dependencies.Input("3d-scatter-plot", "hoverData"), dash.dependencies.Input("sample-ratio", "value")]

    )
    def ugdate_figure(hover_data_from_3d_scatter, sample_sizes):
        try:
            hoverData = hover_data_from_3d_scatter['points'][0]['PaymentTransactionId']
        except:
            hoverData = hover_data_from_3d_scatter['points'][0]['customdata']

        dff = samples_dict[sample_sizes]['data'].query("PaymentTransactionId == @hoverData")
        trace = [go.Table(
            header=dict(values=related_columns,
                        line_color='darkslategray',
                        fill_color='lightcyan',
                        align='left'),
            cells=dict(values=[dff[col] for col in related_columns],
                       line_color='darkslategray',
                       fill_color='white',
                       align='left'))]
        return {"data": trace,
                'layout': go.Layout(height=400, title=' customers Transactions')
                }

        # Histogram

    @app.callback(
        dash.dependencies.Output("my-graph_3", "figure"),
        [dash.dependencies.Input("3d-scatter-plot", "hoverData"),
         dash.dependencies.Input("model-selection", "value"),
         dash.dependencies.Input("sample-ratio", "value")]

    )
    def get_c_freq_diff(hover_data_from_3d_scatter, model_selection, sample_sizes):
        try:
            hoverData = hover_data_from_3d_scatter['points'][0]['PaymentTransactionId']
        except:
            hoverData = hover_data_from_3d_scatter['points'][0]['customdata']
        print("TRANSACTION :::", hoverData)
        customer_merchant_id = list(
            samples_dict[sample_sizes]['data'][['PaymentTransactionId', 'customer_merchant_id']].query(
                "PaymentTransactionId == @hoverData")['customer_merchant_id'])[0]

        dff = \
        samples_dict[sample_sizes]['customer_transactions'].query("customer_merchant_id == @customer_merchant_id")[
            ['c_freq_diff', 'RequestInsertTime', 'label_iso']]
        # dff = pd.DataFrame(zip(customers_of_amounts[customer_merchant_id]['Amount'], customers_of_amounts[customer_merchant_id]['RequestInsertTime']))
        print(customer_merchant_id.split("_")[0])
        return {"data": [go.Scatter(x=dff['RequestInsertTime'], y=dff['c_freq_diff'], mode='markers')],
                "layout": go.Layout(height=300,
                                    title=features_cols_2['c_freq_diff'] + " || C:" + customer_merchant_id.split("_")[0])
                }
    if is_local_run:
        webbrowser.open('http://127.0.0.1:8050/')
        app.run_server(debug=False)
    else:
        app_is_open = True
        while not app_is_open:
            app.run_server(debug=False, port=3030, host='10.20.10.196')
            app_is_open = False