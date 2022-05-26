#!/usr/bin/env python
# coding: utf-8

import time
import multiprocess
from math import log
import plotly.express as px
import numpy as np
import time
import multiprocessing
from PIL import Image

def escape_count2(z,c, max_iterations, escape_radius):

    for num in range(max_iterations):
        z = z ** 2 + c
        if abs(z) > escape_radius:
            return num + 1 - log(log(abs(z))) / log(2)
    return max_iterations

def stability2(z,c,max_iterations, escape_radius):
    value = float(escape_count2(z,c,max_iterations, escape_radius)) / float(max_iterations)
    return  max(0.0, min(value, 1.0))


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


def image_func(arg_list):

    image  =arg_list[0] 
    start_height=arg_list[1] 
    end_height=arg_list[2] 
    width=arg_list[3]
    move_x=arg_list[4]
    move_y=arg_list[5]
    max_iterations=arg_list[6]
    escape_radius=arg_list[7]
    scale=arg_list[8]
    for y in range(start_height,end_height):
        for x in range(width):
            c = scale * complex((x +move_x) - width / 2, 4*(end_height-start_height) / 2 - (y + move_y) )
            instability = 1 - stability(c,max_iterations, escape_radius)
            image.putpixel((x, y-start_height), int(instability * 255)) 
    return image


def image_func_julia(arg_list):

    image  =arg_list[0] 
    start_height=arg_list[1] 
    end_height=arg_list[2] 
    width=arg_list[3]
    move_x=arg_list[4]
    move_y=arg_list[5]
    max_iterations=arg_list[6]
    escape_radius=arg_list[7]
    scale=arg_list[8]
    c=arg_list[9]
    scale2=arg_list[10]


    for y in range(start_height,end_height):
        for x in range(width):
            z = scale2 * complex((x) - width / 2, 4*(end_height-start_height) / 2 - (y) )
            

            instability = 1 - stability2(z,c,max_iterations, escape_radius)
            image.putpixel((x, y-start_height), int(instability * 255))  
    return image