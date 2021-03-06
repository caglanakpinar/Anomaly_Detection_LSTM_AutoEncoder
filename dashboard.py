import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import random
import plotly.graph_objs as go
import webbrowser

from configs import sample_sizes, k_means_cluster_colors, related_cols, feature_path, host, port
from configs import models_output, related_columns, is_local_run, learning_model_path
from data_access import decide_feature_name, model_from_to_json


def get_bias_condition_for_sampling(data, models):
    bias_query_str = ''
    for label in models:
        if label in ['recon_ad', 'anomaly_ae_values']:
            treshold = list(data[label])[int(len(data) * 0.01)]
            data[label] = data[label].apply(lambda x: 0 if x < treshold else -1)
            if treshold == treshold:
                bias_query_str += label + " > " + str(treshold) + " or "
    return bias_query_str, data

def get_sample_from_data(main_data, sample_ratio, bias_data_condition=None):
    main_data = main_data.reset_index(drop=True)
    bias_index = list(main_data.query(bias_data_condition).reset_index()['index']) if bias_data_condition is not None else []
    sample_size = int(len(main_data) * sample_ratio)
    random_index = set(random.sample(list(range(len(main_data))), sample_size) + bias_index)
    indexes = list(main_data.index)
    return main_data.ix[list(set(random_index) & set(main_data.index))].reset_index()


def dashboard_init():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    return dash.Dash(__name__, external_stylesheets=external_stylesheets)


def create_dahboard(df_train, df):
    fea_dict = decide_feature_name(feature_path)
    feature = list(fea_dict.keys())
    model_dict = model_from_to_json(learning_model_path, [], False)
    models_output = {model_dict[m]['args']['pred_field']: model_dict[m]['name'] for m in model_dict}
    for m in models_output:
        df_train[m] = 0
    df_train = pd.concat([df_train[related_cols + feature + list(models_output.keys())].reset_index(drop=True),
                          df[related_cols].reset_index(drop=True)]).query("Amount == Amount")

    customer_merchant_ratios = df_train.pivot_table(index=['customer_id', 'merchant_id'],
                                                    aggfunc={'c_m_label_t_count': 'max', 'c_m_t_count': 'max',
                                                             'label_iso': 'sum'}).reset_index()
    customer_merchant_ratios['merchant_id'] = customer_merchant_ratios['merchant_id'].apply(lambda x: str(x))
    bias_query_str, df = get_bias_condition_for_sampling(df, models_output)

    samples_dict = {}
    for s in sample_sizes:
        samples_dict[s[0]] = {'data': get_sample_from_data(df, s[1], bias_query_str + "slope > 0")}
        _customers = list(samples_dict[s[0]]['data']['customer_id'].unique())
        samples_dict[s[0]]['c_m_ratios'] = customer_merchant_ratios.query("customer_id in @_customers")
        samples_dict[s[0]]['customer_transactions'] = df_train.query("customer_id in @_customers")
    _customers = list(df.sort_values(by='anomaly_ae_values', ascending=False)['customer_id'].unique())[0:100]
    merchants = list(df.sort_values(by='anomaly_ae_values', ascending=False)['merchant_id'].unique())[0:100]

    app = dashboard_init()
    app.layout = html.Div([
        html.Div([html.H1("Anomaly Detection Multivariate Isolation Foreset - AutoEncoder")],
                 style={'textAlign': "left", "padding-bottom": "10", "padding-top": "10"}),
        html.Div(
            [html.Div(dcc.Dropdown(id="select-xaxis",
                                   options=[{'label': fea_dict[i]['name'], 'value': i} for i in fea_dict],
                                   value=feature[0], ), className="four columns",
                      style={"display": "block", "margin-left": "!%",
                             "margin-right": "auto", "width": "33%"}),
             html.Div(dcc.Dropdown(id="select-yaxis",
                                   options=[{'label': fea_dict[i]['name'], 'value': i} for i in fea_dict],
                                   value=feature[1], ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "auto", "width": "33%"}),
             html.Div(dcc.Dropdown(id="select-zaxis",
                                   options=[{'label': fea_dict[i]['name'], 'value': i} for i in fea_dict],
                                   value=feature[2], ), className="four columns",
                      style={"display": "block", "margin-left": "auto",
                             "margin-right": "1%", "width": "33%"})
             ], className="row", style={"padding": 14, "display": "block", "margin-left": "1%",
                                        "margin-right": "auto", "width": "99%"}),
        html.Div(
            [html.Div(dcc.Dropdown(id="model-selection", options=[{'label': models_output[i], 'value': i} for i in
                                                                  models_output.keys()],
                                   value='label_iso', ), className="four columns",
                      style={"display": "block", "margin-left": "1%",
                             "margin-right": "auto", "width": "49%"}),
             html.Div(dcc.Dropdown(id="sample-ratio", options=[{'label': i[0].title(), 'value': i[0]} for i in
                                                               sample_sizes],
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
        if customer != 'ALL':
            dff = samples_dict[sample_sizes]['data'].query("customer_id == @customer")
        else:
            dff = samples_dict[sample_sizes]['data']
        if merchant != 'ALL':
            dff = dff.query("merchant_id == @merchant")
        print(dff.columns)
        color = dff[model_selection]
        trace = [go.Scatter3d(
                              x=dff[selected_x],
                              y=dff[selected_y],
                              z=dff[selected_z],
                              customdata=dff['PaymentTransactionId'],
                              mode='markers',
                              marker={'size': 3, 'color': color,
                                      'colorscale': k_means_cluster_colors,
                                      'opacity': 0.8, "showscale": False,
                                      "colorbar": {"thickness": 10, "len": 0.5, "x": 0.8, "y": 0.6, }, })]

        return {"data": trace,
                "layout": go.Layout(
                    height=700,
                    hovermode='closest',
                    scene={"aspectmode": "cube", "xaxis": {"title": f"{fea_dict[selected_x]['name'].title()}", },
                           "yaxis": {"title": f"{fea_dict[selected_y]['name'].title()}", },
                           "zaxis": {"title": f"{fea_dict[selected_z]['name'].title()}", }})
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
                                    title=fea_dict['c_m_ratios']['name'] + " || Card: " + customer_id,
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

        dff = df_train.query("customer_merchant_id == @customer_merchant_id")[['Amount', 'RequestInsertTime', 'label_iso']]
        return {"data": [go.Scatter(x=dff['RequestInsertTime'], y=dff['Amount'], mode='markers')],
                "layout": go.Layout(height=300,
                                    title=fea_dict['c_m_peak_drop_min_max_p_value']['name'] + " || C_M:" + customer_merchant_id if customer_merchant_id is not None else " - "
                                    )
                }

    # table
    @app.callback(
        dash.dependencies.Output("my-graph_2", "figure"),
        [dash.dependencies.Input("3d-scatter-plot", "hoverData"), dash.dependencies.Input("sample-ratio", "value")]

    )
    def ugdate_figure(hover_data_from_3d_scatter, sample_sizes):
        try:
            hoverData = hover_data_from_3d_scatter['points'][0]['PaymentTran    sactionId']
        except:
            hoverData = hover_data_from_3d_scatter['points'][0]['customdata']

        dff = samples_dict[sample_sizes]['data'].query("PaymentTransactionId == @hoverData")
        trace = [go.Table(
            header=dict(values=related_columns + feature,
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
         dash.dependencies.Input("sample-ratio", "value")]
    )
    def get_c_freq_diff(hover_data_from_3d_scatter, sample_sizes):
        try:
            hoverData = hover_data_from_3d_scatter['points'][0]['PaymentTransactionId']
        except:
            hoverData = hover_data_from_3d_scatter['points'][0]['customdata']
        print("TRANSACTION :::", hoverData)
        customer_merchant_id = list(
            samples_dict[sample_sizes]['data'][['PaymentTransactionId', 'customer_merchant_id']].query(
                "PaymentTransactionId == @hoverData")['customer_merchant_id'])[0]

        dff = samples_dict[sample_sizes]['customer_transactions'].query("customer_merchant_id == @customer_merchant_id")[
            ['c_freq_diff', 'RequestInsertTime', 'label_iso']]
        print(customer_merchant_id)
        return {"data": [go.Scatter(x=dff['RequestInsertTime'], y=dff['c_freq_diff'], mode='markers')],
                "layout": go.Layout(height=300,
                                    title="C. Difference Of Each Transaction Score")
                }
    if is_local_run:
        webbrowser.open('http://127.0.0.1:8050/')
        app.run_server(debug=False)
    else:
        webbrowser.open('http://127.0.0.1:8050/')
        return app.run_server(debug=False)# app.run_server(debug=False, port=port, host=host)