#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:20:08 2025

Utilities for the NTPN Application


@author: proxy_loken
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import skimage.io as skio
from PIL import Image
import time

from ntpn import point_net_utils
from ntpn import ntpn_constants
from ntpn import point_net

import tensorflow as tf
from tensorflow import keras

def initialise_session():
    
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = ntpn_constants.dataset_name
        load_demo_session()
    
    #initialise necessary variables
    st.session_state.ntpn_model = 0
    
    
    return


# LOADING/SAVING UTILITIES

def load_demo_session():
    
    # files for dataset and labels
    st_file = ntpn_constants.demo_st_file
    context_file = ntpn_constants.demo_context_file

    # load dataset and labels
    stbin_list, context_list = point_net_utils.load_data_pickle(st_file, context_file, 'context_labels')
    
    
    st.session_state.dataset = stbin_list
    st.session_state.labels = context_list
    
        
    return


# loading for features by samples matrices as input data
def load_2D_data():
    
    
    return


# loading for input data that is pre-sliced into trajectories
def load_3D_data():
    
    
    return


# DATASET PROCESSING

def session_select(sessions, trim_noise):
    select_samples = [st.session_state.dataset[i] for i in sessions]
    select_labels = [st.session_state.labels[i] for i in sessions]    
    select_indices = [(idx) for idx, item in enumerate(select_samples)]
    if trim_noise:
        select_samples, select_labels = point_net_utils.remove_noise_cat(select_samples, select_labels, select_indices)
    
    st.session_state.select_samples = select_samples
    st.session_state.select_labels = select_labels
    st.session_state.select_indices = select_indices   
    return


def samples_transform(transform_radio):
    
    if transform_radio=='Power':
        X_tsf = point_net_utils.pow_transform(st.session_state.select_samples, st.session.select_indices)
    elif transform_radio=='Standard':
        X_tsf = point_net_utils.std_transform(st.session_state.select_samples, st.session.select_indices)
    else:
        X_tsf = st.session_state.select_samples
        
    st.session_state.tsf_samples = X_tsf    
    return


def create_trajectories(trajectories_window_size, trajectories_window_stride, trajectories_num_neurons):
    
    
    # Project into 3D via sliding windows
    X_sw_list, Y_sw_list = point_net_utils.window_projection(st.session_state.tsf_samples, st.session_state.select_labels, st.session_state.select_indices, window_size=trajectories_window_size, stride=trajectories_window_stride)
    # Within Session Dataset Gen
    X_subs, Ys = point_net_utils.subsample_dataset_3d_within(X_sw_list, Y_sw_list, trajectories_num_neurons, replace=False)
    X_subs = np.swapaxes(X_subs,1,2)
    
    st.session_state.sub_samples = X_subs
    st.session_state.sub_labels = Ys
    return



def create_train_test(test_size):
    
    # Make training and Test sets
    X_train, X_val, Y_train, Y_val = point_net_utils.train_test_gen(st.session_state.sub_samples, st.session_state.sub_labels, test_size=test_size)
    # Make tensors for the point net
    train_dataset, test_dataset = point_net_utils.train_test_tensors(X_train, X_val, Y_train, Y_val, augment=False)
    
    st.session_state.train_tensors = train_dataset
    st.session_state.test_tensors = test_dataset    
    return

# NTPN MODEL UTILITIES

# create a ntpn model by calling on the point net class
def create_model(trajectory_length, num_classes, layer_width, trajectory_dim):
    
    st.session_state.ntpn_model = point_net.point_net(trajectory_length, num_classes, units=layer_width, dims=trajectory_dim)
        
    return

# compile the model; single line if not viewing training (built-in keras fit), manually if usin the streamlit version
def compile_model(loss='sparse_categorical_crossentropy', learning_rate=0.02, metric='sparse_categorical_accuracy', view=True):
    
    # TODO: refactor to account for possible different loss functions, metrics, etc. 
    
    if view:
        st.session_state.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        st.session_state.train_metric = keras.metrics.SparseCategoricalAccuracy()
        st.session_state.test_metric = keras.metrics.SparseCategoricalAccuracy()
        st.session_state.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    st.session_state.ntpn_model.compile(loss=loss,optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=[metric])
        
    return

# replaces keras fit method to enable progress to be displayed inside streamlit
# adapted from shubhadtiya goswami
def train_for_streamlit(epochs):
    #TODO: remove this, add batch size to dataset params
    st.session_state.batch_size = 8    
    
    st.write('Starting Training with {} epochs...'.format(epochs))
    
    for epoch in range(epochs):
        st.write("Epoch {}".format(epoch+1))
        start_time = time.time()
        #initialise display params
        progress_bar = st.progress(value=0.0)
        percent_complete = 0
        epoch_time = 0
        # placeholder for update step
        st_t = st.empty()
        
        train_loss_list = []
        # Iterate over batches
        for step, (x_batch_train, y_batch_train) in enumerate(st.session_state.train_tensors):
            start_step = time.time()
            loss_value = train_step(x_batch_train,y_batch_train)
            end_step = time.time()
            epoch_time+= (end_step - start_step)
            train_loss_list.append(float(loss_value))       
        
            # number of steps to log
            if step % 1 == 0:
                step_acc = float(st.session_state.train_metric.result())
                percent_complete = ((step/(len(st.session_state.sub_samples)//st.session_state.batch_size)))
                progress_bar.progress(percent_complete)
                st_t.write('Duration : {0:.2f}s, Training Acc : {1:.4f}'.format((epoch_time),float(step_acc)))
            
        progress_bar.progress(1.0)
        
        # Metrics for the end of each epoch
        train_acc = st.session_state.train_metric.result()
        # reset training metric at the end of each epoch
        st.session_state.train_metric.reset_state()
        
        train_loss = round((sum(train_loss_list)/len(train_loss_list)),5)
        
        val_loss_list = []
        # run the validation loop
        for x_batch_val, y_batch_val in st.session_state.test_tensors:
            val_loss_list.append(float(test_step(x_batch_val, y_batch_val)))
            
        val_loss = round((sum(val_loss_list)/len(val_loss_list)),5)
        
        val_acc = st.session_state.test_metric.result()
        st.session_state.test_metric.reset_state()
        
        st_t.write('Duration : {0:.2f}s, Training Acc : {1:.4f}, Validation Acc : {2:.4f}'.format((time.time() - start_time),float(train_acc), float(val_acc)))
        
           
    return


def train_model(epochs, view=True):
    
    if view:
        train_for_streamlit(epochs)
    else:    
        st.session_state.ntpn_model.fit(st.session_state.train_tensors, epochs=epochs, validation_data=st.session_state.test_tensors)
        
    return


# NTPN MODEL TRAINING AND TESTING FUNCTIONS (necessary for display in streamlit)

# gradient, loss, and metric calculation for the model and data
# adapted from shubhadtiya goswami
def train_step(x, y):
    
    model = st.session_state.ntpn_model
    with tf.GradientTape() as tape:
        #predictions = st.session_state.ntpn_model(x, training=True)
        predictions = model(x, training=True)
        loss_value = st.session_state.loss_fn(y, predictions)
        #loss_value = model.compute_loss(y, predictions)
    # calculate gradients
    #grads = tape.gradient(loss_value, st.session_state.ntpn_model.trainable_weights)
    grads = tape.gradient(loss_value, model.trainable_weights)
    # apply gradients via optimizer
    #st.session_state.optimizer.apply_gradients(zip(grads, st.session_state.ntpn_model.trainable_weights))
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # update loss metric    
    for metric in model.metrics:
        if metric.name == 'loss':
            metric.update_state(loss_value)
        else:
            metric.update_state(y, predictions)
    st.session_state.train_metric.update_state(y, predictions)
    return loss_value


def test_step(x, y):
    
    val_predictions = st.session_state.ntpn_model(x, training=False)
    st.session_state.test_metric.update_state(y, val_predictions)
    return st.session_state.loss_fn(y, val_predictions)




# helper to export model as a .h5(keras) file for re-use
def save_model(model_name):
    
    st.session_state.ntpn_model.save('models/'+model_name+'.keras', overwrite=True, include_optimizer=True)
    
    
    return


# CRITICAL SETS AND UPPER-BOUND Functions

# generate critical sets for a number of classes and samples 
# NOTE: may need to explicitly name the correct activation layer(before the pooling operation) during model creation if multiple models breaks this function
def generate_critical_sets(num_classes, num_samples):
    
    cs_trajectories = []
    cs_predictions = []
    cs_lists = []
    cs_means = []
    for i in range(num_classes):
        class_trajectories = point_net_utils.select_samples(st.session_state.sub_samples, st.session_state.sub_labels, num_samples, i)
        cs_trajectories.append(class_trajectories)
        class_predictions = point_net.predict_critical(st.session_state.ntpn_model, class_trajectories, layer_name = 'activation_14')
        cs_predictions.append(class_predictions)
        class_cs, class_cs_mean = point_net.generate_critical(class_predictions, num_samples, class_trajectories)
        cs_lists.append(class_cs)
        cs_means.append(class_cs_mean)
    
    st.session_state.cs_trajectories = cs_trajectories
    st.session_state.cs_predictions = cs_predictions
    st.session_state.cs_lists = cs_lists
    st.session_state.cs_means = cs_means
        
    return



def cs_downsample_PCA(label, num_examples, dims=3):
    
    pca_cs, pca_trajs = point_net_utils.pca_cs_windowed(st.session_state.cs_lists[label], st.session_state.cs_trajectories[label], dims=dims)
    
    pca_css, pca_trajss = point_net_utils.select_samples_cs(pca_cs, pca_trajs, num_examples)
        
    return pca_css, pca_trajss


def cs_downsample_UMAP(label, num_examples, dims=3):
    
    umap_cs, umap_trajs = point_net_utils.umap_cs_windowed(st.session_state.cs_lists[label], st.session_state.cs_trajectories[label], dims=dims)
    
    umap_css, umap_trajss = point_net_utils.select_samples_cs(umap_cs, umap_trajs, num_examples)
    
    return umap_css, umap_trajs


def cs_CCA_alignment():
    
    
    return



def plot_trajectories_UMAP():
    
    return


def plot_critical_sets_PCA(label, num_examples, dims=3):
    
    pca_css, pca_trajss = cs_downsample_PCA(label, num_examples, dims=dims)
    
    fig = point_net_utils.plot_critical(pca_css, num_examples, pca_trajss)
    
    return fig


def plot_critical_sets_UMAP(label, num_examples, dims=3):
    
    umap_css, umap_trajss = cs_downsample_UMAP(label, num_examples, dims=dims)
    
    fig = point_net_utils.plot_critical(umap_css, num_examples, umap_trajss)
    
    return fig


def plot_critical_sets_grid():    
    
    return


# DRAWING UTILITIES

def draw_image(image, header, description):
    
    # Draw header and image
    st.subheader(header)
    st.markdown(description)
    st.image(image.astype(np.uint8), use_container_width=True)    
    return


def draw_cs_plots(plotting_algo, num_examples, dims, num_classes):
    
    figs = []
    if plotting_algo == 'PCA':
        for i in range(num_classes):
            fig = plot_critical_sets_PCA(i, num_examples, dims)
            figs.append(fig)
    elif plotting_algo == 'UMAP':
        for i in range(num_classes):
            fig = plot_critical_sets_UMAP(i, num_examples, dims)
            figs.append(fig)
    
    st.session_state.cs_ub_plots = figs
    
    return




