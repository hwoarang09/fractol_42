#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import log

def escape_count(c, max_iterations, escape_radius):
    z = 0
    for num in range(max_iterations):
        z = z ** 2 + c
        if abs(z) > escape_radius:
            return num + 1 - log(log(abs(z))) / log(2)
    return max_iterations

def stability(c,max_iterations, escape_radius):
    value = float(escape_count(c,max_iterations, escape_radius)) / float(max_iterations)
    return  max(0.0, min(value, 1.0))
def escape_count2(z,c, max_iterations, escape_radius):

    for num in range(max_iterations):
        z = z ** 2 + c
        if abs(z) > escape_radius:
            return num + 1 - log(log(abs(z))) / log(2)
    return max_iterations

def stability2(z,c,max_iterations, escape_radius):
    value = float(escape_count2(z,c,max_iterations, escape_radius)) / float(max_iterations)
    return  max(0.0, min(value, 1.0))


# In[7]:


import dash
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
import numpy as np
import json
import time
import multiprocess
from image_func import image_func
width, height = 512, 512


scale_default = 0.0078
scale = scale_default
GRAYSCALE = "L"
max_iterations = 26
escape_radius = 10
move_x = -0.7435 / scale
move_y = 0.1314 / scale

from PIL import Image

np.warnings.filterwarnings("ignore")

app = dash.Dash()

app.layout = html.Div([
    
    html.Div([
    
    
    html.Div([
            html.Div(children=[dcc.Graph(id='graph')]),
        
            html.Div([
                html.Div([
                          html.H3('Escape_radius',
                            style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='escape_radius', type='number', debounce=True, min=2, max=1000,step=1, value = 10,
                            style={'width' : '50px', 'margin-left' : '10px', 'margin-right' : '30px','display' : 'inline-block', 'float' : 'left'})
                         ],style = {'display' : 'inline-block','float':'left'}),
                html.Div([
                          html.H3('Max_iterations ',
                            style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='max_iterations', type='number', debounce=True, min=2, max=1250,step=1, value = 26,
                            style={'width' : '50px', 'margin-left' : '10px','display' : 'inline-block', 'float' : 'left'})
                         ],style = {'display' : 'inline-block','float':'left'})
            ], style = {'display' : 'flex', 'float':'none'}),
        
            html.Div([
                html.Div([
                          html.H3('x center ',
                            style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='move_x', type='text', value = -0.7435,
                            style={'width' : '50px', 'margin-left' : '60px', 'margin-right' : '30px','display' : 'inline-block', 'float' : 'left'})         
                         ],style = {'display' : 'inline-block','float':'left'}),           
                html.Div([
                          html.H3('y center ',
                           style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='move_y', type='text', value =0.1314,
                            style={'width' : '50px', 'margin-left' : '70px','display' : 'inline-block', 'float' : 'left'})            
                         ],style = {'display' : 'inline-block','float':'left'})       
            ], style = {'display' : 'flex', 'float':'none'}),
            html.Div([
                html.Div([
                          html.H3('scale ',
                            style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='scale', type='number', value =1,
                            style={'width' : '50px', 'margin-left' : '87px', 'margin-right' : '30px','display' : 'inline-block', 'float' : 'left'})                
                         ],style = {'display' : 'inline-block','float':'left'}),         
                html.Pre(id = 'hover-data',style = {'display' : 'inline-block','float':'left'})
            ], style = {'display' : 'flex', 'float': 'none'}),
        
            html.Div([
            html.Button(id='submit-button-state', n_clicks=0, children='Submit',
                    style={'textAlign': 'center','color': 'black','display':'inline-block', 'float' : 'left'})
            ], style = {'display' : 'flex', 'float':'none'})
            ],
             
        style={'textAlign': 'center','color': 'black', 'display':'inline-block', 'width' : '50%', 'float' : 'left'}),

        
    html.Div([
            html.Div(children=[dcc.Graph(id='graph2')]),
        
            html.Div([
                html.Div([
                          html.H3('Escape_radius',
                            style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='escape_radius2', type='number', debounce=True, min=2, step=1, value = 10,
                            style={'width' : '50px', 'margin-left' : '10px', 'margin-right' : '30px','display' : 'inline-block', 'float' : 'left'})
                         ],style = {'display' : 'inline-block','float':'left'}),
                html.Div([
                          html.H3('Max_iterations ',
                            style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='max_iterations2', type='number', debounce=True, min=2, step=1, value = 26,
                            style={'width' : '50px', 'margin-left' : '10px','display' : 'inline-block', 'float' : 'left'})
                         ],style = {'display' : 'inline-block','float':'left'})
            ], style = {'display' : 'flex', 'float':'none'}),
        
            html.Div([
                html.Div([
                          html.H3('x center ',
                            style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='move_x2', type='text', value = 0,
                            style={'width' : '50px', 'margin-left' : '60px', 'margin-right' : '30px','display' : 'inline-block', 'float' : 'left'})         
                         ],style = {'display' : 'inline-block','float':'left'}),           
                html.Div([
                          html.H3('y center ',
                           style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='move_y2', type='text', value =0,
                            style={'width' : '50px', 'margin-left' : '70px','display' : 'inline-block', 'float' : 'left'})            
                         ],style = {'display' : 'inline-block','float':'left'})       
            ], style = {'display' : 'flex', 'float':'none'}),
            html.Div([
                html.Div([
                          html.H3('scale ',
                            style={'margin' : '0','color': 'black','display':'inline-block', 'float' : 'left'}),
                          dcc.Input(id='scale2', type='number', value =1,
                            style={'width' : '50px', 'margin-left' : '87px', 'margin-right' : '30px','display' : 'inline-block', 'float' : 'left'})                
                         ],style = {'display' : 'inline-block','float':'left'})    
                
            ], style = {'display' : 'flex', 'float': 'none'})
        
            
            ],
             
        style={'textAlign': 'center','color': 'black', 'display':'inline-block', 'width' : '50%', 'float' : 'left'}),

], style={'textAlign': 'center','color': 'black', 'display':'inline-block', 'width' : '70%'}),
    html.Div([
            html.Div([
                html.H6('julia set 아래 메뉴들은 미구현입니다',
                    style={'color': 'black'}),
                html.H3('-사용법- ',
                    style={'color': 'black'}),                
                html.H5('1. mandelbrot의 scale을 10정도로 입력해보고, submit을 눌러보세요 ',
                    style={'color': 'black'}),
                html.H5('2. 경계에서 명확하게 프랙탈이 보이지 않으면, Max_iterations을 50정도 입력해보고 submit을 눌러보세요 ',
                    style={'color': 'black'}),
                html.H5('3. x center, y center를 변경하면서 그림을 옮길 수 있어요',
                    style={'color': 'black'}),        
                html.H5('4. mandelbrot fractal 그래프를 클릭하면, c값이 정해집니다.',
                    style={'color': 'black'}),    
                html.H5('5. 동시에 우측에서는 c값을 기준으로 julia set이 그려져요. ',
                    style={'color': 'black'}),
                html.H3('-그래프메뉴-',
                    style={'color': 'black'}),                
                html.H5('1. 그래프에 마우스를 가져다 대면 상단에 여러 메뉴들이 보입니다. 확대, 축소, 이동, 원상복귀 등등',
                    style={'color': 'black'}),
                html.H5('2. 그래프를 드래그해도 확대가 되지만, 올바른 c값 및 julia set은 생성되지 않습니다.',
                    style={'color': 'black'}),                
            ],  style={'textAlign': 'left','color': 'black', 'display':'inline-block', 'width' : '70%'})
    ])
]
    ,style={'textAlign': 'center'})
@app.callback(Output('hover-data', 'children'),
              Input('graph', 'hoverData'),
              State('escape_radius', 'value'),
              State('max_iterations', 'value'),
              State('move_x', 'value'),
              State('move_y', 'value'),
              State('scale', 'value'))

def dis_play_hover_data(hover_data,escape_radius, max_iterations,move_x,move_y,scale):
    if hover_data == None:
        return f'복소수 c : {0}+{0}i '
    scale_default = 0.0078
    scale = scale_default / scale    
    raw = json.dumps(hover_data, indent=2)
    x = hover_data["points"][0]["x"]
    y = hover_data["points"][0]["y"]
    move_x = float(move_x) / scale
    move_y = float(move_y) / scale    
    c = scale * complex((x +move_x) - width / 2, height / 2 - (y + move_y) )
    
    print(type(hover_data),hover_data, hover_data["points"])
    return f'복소수 c : {c.real:{".4f"}}+{c.imag:{".4f"}}i '

########################################figure hover click###############

@app.callback(Output('graph2', 'figure'),
              Input('graph', 'clickData'),
              State('escape_radius', 'value'),
              State('max_iterations', 'value'),
              State('move_x', 'value'),
              State('move_y', 'value'),
              State('scale', 'value'),
              State('escape_radius2', 'value'),
              State('max_iterations2', 'value'),
              State('move_x2', 'value'),
              State('move_y2', 'value'),
              State('scale2', 'value'))
def update_figure(clickData,escape_radius,max_iterations,move_x,move_y,scale,
                 escape_radius2,max_iterations2,move_x2,move_y2,scale2):

    

    
    if clickData is None:
        scale_default = 0.0078
        scale = scale_default / scale
        GRAYSCALE = "L"
        cx = 0
        cy = 0
        move_x = float(move_x) / scale
        move_y = float(move_y) / scale

        image = Image.new(mode=GRAYSCALE, size=(width, height))
        c = complex(0,0)
        
        for y in range(height):
            for x in range(width):
                z = scale * complex((x) - width / 2, height / 2 - (y) )
                instability = 1 - stability2(z,c,max_iterations, escape_radius)
                image.putpixel((x, y), int(instability * 255))        


        fig = px.imshow(image,color_continuous_scale="Rainbow",range_color=[0,255],
                       title="Julia Sets Fractal")      
        fig.update_traces(hovertemplate="x: %{x} <br> y: %{y}<extra></extra>")
        fig.update(layout_coloraxis_showscale=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)    
        fig.update_layout(margin_b=50, margin_l = 0, margin_r=0, margin_t = 50)
        return fig
  
    else:
 
        scale_default = 0.0078
        scale = scale_default / scale
        scale2 = scale_default / scale2
        GRAYSCALE = "L"
        cx = clickData["points"][0]["x"]
        cy = clickData["points"][0]["y"]
        move_x_c = float(move_x) / scale
        move_y_c = float(move_y) / scale
        c = scale * complex((cx+move_x_c) - width / 2, height / 2 - (cy+move_y_c) )
        image = Image.new(mode=GRAYSCALE, size=(width, height))

        
        for y in range(height):
            for x in range(width):
                z = scale2 * complex((x) - width / 2, height / 2 - (y) )
                instability = 1 - stability2(z,c,max_iterations, escape_radius)
                image.putpixel((x, y), int(instability * 255))        

        fig = px.imshow(image,color_continuous_scale="Rainbow",range_color=[0,255],
                       title="Julia Sets Fractal")    
        fig.update_traces(hovertemplate="x: %{x} <br> y: %{y}<extra></extra>")
        fig.update(layout_coloraxis_showscale=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(margin_b=50, margin_l = 0, margin_r=0, margin_t = 50)
        return fig



##########################################################################################
@app.callback(Output('graph', 'figure'),
              Input('submit-button-state', 'n_clicks'),
              State('escape_radius', 'value'),
              State('max_iterations', 'value'),
              State('move_x', 'value'),
              State('move_y', 'value'),
              State('scale', 'value'))
def update_figure(n_clicks,escape_radius,max_iterations,move_x,move_y,scale):
    print("typeof movex ", type(move_x))
    scale_default = 0.0078
    scale = scale_default / scale
    GRAYSCALE = "L"

    move_x = float(move_x) / scale
    move_y = float(move_y) / scale

    

    

    image = Image.new(mode=GRAYSCALE, size=(width, height))
    image1 = Image.new(mode=GRAYSCALE, size=(width, int(height/4)))
    image2 = Image.new(mode=GRAYSCALE, size=(width, int(height/4)))
    image3 = Image.new(mode=GRAYSCALE, size=(width, int(height/4)))
    image4 = Image.new(mode=GRAYSCALE, size=(width, int(height/4)))

    arg_list = [
        [image1, 0, int(height*(1/4)), width, move_x, move_y, max_iterations, escape_radius,scale],
        [image2, int(height*(1/4)),int(height*(2/4)), width, move_x, move_y, max_iterations, escape_radius,scale],
        [image3, int(height*(2/4)),int(height*(3/4)), width, move_x, move_y, max_iterations, escape_radius,scale],
        [image4, int(height*(3/4)),int(height*(4/4)), width, move_x, move_y, max_iterations, escape_radius,scale]
    ]

    start = time.time()
    pool = multiprocess.Pool(processes = 4)
    job=(pool.map(image_func, arg_list))
    pool.close()
    pool.join()
    print(f'----after pool {time.time()-start} seconds -----')

    image_sum = Image.new(mode=GRAYSCALE, size=(width, height))
    image_sum.paste(job[0], (0,0))
    image_sum.paste(job[1], (0,128))
    image_sum.paste(job[2], (0,256))
    image_sum.paste(job[3], (0,384))
    '''
    image = Image.new(mode=GRAYSCALE, size=(width, height))
    for y in range(height):
        for x in range(width):
            c = scale * complex((x +move_x) - width / 2, height / 2 - (y + move_y) )
            instability = 1 - stability(c,max_iterations, escape_radius)
            image.putpixel((x, y), int(instability * 255))
    '''                
    fig = px.imshow(image_sum,color_continuous_scale="RdGy",range_color=[0,255],
                   title="Mandelbrot Sets Fractal")    
      
    ##fig.update_traces(hovertemplate="x: %{x / scale} <br> y: %{float(y)}<extra></extra>")
    fig.update_traces(hovertemplate="x: %{x} <br> y: %{y}<extra></extra>")
    fig.update(layout_coloraxis_showscale=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)    
    fig.update_layout(margin_b=50, margin_l = 0, margin_r=0, margin_t = 50)
    print(f'----before fig {time.time()-start} seconds -----')
    return fig


if __name__ == '__main__':
    app.run_server()


# In[ ]:




