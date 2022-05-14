import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import helper as hp

from PIL import Image
from tensorflow import keras
import plotly.figure_factory as ff
import plotly.express as px


# set title of app
st.title('Predicting the Probability of Glaucoma Pathology Using a Convolutional Neural Network')

# establish pages
with st.sidebar:
    st.image('../data/test_images/LA_left_eye.jpg', width=300)    
    page = st.sidebar.selectbox(
        'Select a page:',
        ('Home', 
         'About',
         'Make a prediction')
    )

# -------------------------------------------------------------------------------------------------------
# if page is 'About' -------------------------------------------------------------------------------------
if page == 'About':
    
    ## DATA -----------------------------------------------------------
    st.header('Convolutional Neural Network for Image Classification')
    st.subheader('''
    __Data:__
    ''')
    
    st.image('../figures/sample_images.png', 
             caption='Fig 1. Sample images of healthy optic disks and those showing Glaucoma pathology.',
            )
    
    col1, col2, col3 = st.columns((2,0.5,2))
    
    with col1:
        st.image('../figures/pixel_distribution.png',
                caption='Fig 2. Distribution of pixel values between 0 and 255 for images of pathalogical (orange) and healthy (blue) optic discs. The dashed lines represent the median value in each distribution.',
                width=400)
    
    with col3:
        st.image('../figures/size_distribution.png',
                caption='Fig 3. Distribution of image heights in pixels of images of pathalogical (orange) and healthy (blue) optic discs. The dashed lines represent the median value in each distribution. All images were resized to 178 x 178 pixels before model training.',
               width=350)
            
    ## MODEL DESCRIPTION -------------------------------------------------
    st.subheader('''
    __Model description:__
    ''')
    st.image('../figures/model_03_auc93.png', 
             caption='Fig 4. A schematic reprentation of the architecture of the CNN optimized for detecting Glaucoma in images of the optic nerve.',
            )
    
    ## MODEL PERFORMANCE ---------------------------------------------------
    st.subheader('''
    __Model Performance:__
    ''')
    col1, col2, col3 = st.columns((4,0.1,2))

    with col1:
        predictions_df = pd.read_csv('../data/test_predictions.csv')

        hist_data = [predictions_df[predictions_df['label']=='Healthy']['preds'], predictions_df[predictions_df['label']=='Glaucoma']['preds']]
        group_labels = ['Healthy', 'Glaucoma']
        colors = ['#1f77b4', '#ff7f0e']

        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)

        # Add title
        fig.update_layout(width=500,
                          height=400,
                          title_text='Distribution of Predicted Probabilities',
                          title_y=0.8,
                          title_x=0.45,
                          plot_bgcolor='white',
                          template='plotly_white',
                          legend=dict(font=dict(size=15)),
                          title_font=dict(size=18))

        fig.update_xaxes(title='Predicted Probability', title_font_size=15, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Density', title_font_size=15, tickfont_size=15, ticks='outside')

        fig.add_vrect(x0=0.3, x1=0.8, 
                      line_width=0, 
                      fillcolor="olivedrab", 
                      opacity=0.2, 
                      annotation=dict(text='threshold range', 
                                      x=0.7, y=0.95, 
                                      bordercolor='olivedrab'))
        # display plot
        st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        st.write('description')

    
    # read dataframe with metric scores at different threshold values
    df = pd.read_csv('../data/threshold_metrics.csv')

    fig = px.line(df, 
                  x='threshold', 
                  y='value', 
                  color='variable',
                  color_discrete_sequence=['#CC6677', '#882255', '#DDCC77', '#44AA99'],
                  line_dash='variable'
                  )

    fig.update_layout(width=600, 
                      height=400,
                      title_text='Model Performance Metrics at different Thresholds for Classification',
                      title_y=0.92,
                      title_x=0.45,
                      plot_bgcolor='white',
                      template='plotly_white',
                      # margin=dict(l=10, r=0, t=60, b=10),
                      # paper_bgcolor="lightsteelblue",
                      legend=dict(font=dict(size=15)),
                      title_font=dict(size=18)
                     )

    # set axes labels
    fig.update_xaxes(title='Threshold value for Perdicted Probability', title_font_size=15, tickfont_size=15, ticks='outside')
    fig.update_yaxes(title='Metric Value', title_font_size=15, tickfont_size=15, ticks='outside')

    fig.update_traces(mode="lines", hovertemplate=None)
    fig.update_layout(hovermode="x")

    # add retangles
    fig.add_vrect(x0=0.3, x1=0.589, line_width=0, fillcolor="palegoldenrod", opacity=0.2, annotation=dict(text='best recall', x=0.58, y=0.96, bordercolor='goldenrod'))
    fig.add_vrect(x0=0.589, x1=0.615, line_width=0, fillcolor="lightslategray", opacity=0.2, annotation=dict(text='best auc', x=0.63, y=0.96, bordercolor='slategrey'))

    # display plot
    st.plotly_chart(fig, use_container_width=True)
    
    z=np.load('../data/confusion_matrix.npy')
    x=['Healthy', 'Glaucoma']
    y=['Healthy', 'Glaucoma']
    z_text = [[str(y) for y in x] for x in z]
    fig = px.imshow(z, x=x, y=y, color_continuous_scale='Viridis', aspect="auto")
    fig.update_traces(
        text=z_text, texttemplate="%{text}", textfont_size=12)
    fig.update_traces(hovertemplate='<br>'.join(['Predicted: %{x}',
                                                 'True: %{y}',
                                                 'Count: %{z}']))

    fig.update_layout(width=400, height=400)
    st.plotly_chart(fig, use_container_width=False)

        

# --------------------------------------------------------------------------------------------------
# if page is 'Make a Prediction' -------------------------------------------------------------------    
if page == 'Make a prediction':
    # set up file uploader 
    st.subheader('Upload an image of the retina using the box below: ')
    uploaded_file = st.file_uploader(label='')
    
    if uploaded_file is not None:
        # preprocess image
        image_data = np.asarray(Image.open(uploaded_file).resize((178,178))) # resize to a size the network can accept and convert to np.array
        image_data = image_data.astype('float32')/255 # normalize pixel values to [0, 1]
        image_data = image_data.reshape(1, 178, 178, 3) # reshape array to appropriate shape for network 
        
        # display uploaded image
        st.subheader('You uploaded this image:')
        st.image(image_data, width=500)
        
        # load saved model         
        model = keras.models.load_model('../models/model_03_auc93/')
        
        # make prediction on image using model
        prediction = model.predict(image_data)[0][0]
        st.write(f"The probability that this retina is pathological for Glaucoma is __{round(prediction*100, 2)}%__. \n The image below shows the distributions of predicted probability of belonging to the class 'Glaucoma' for images of healthy (_orange_) and pathological (_blue_) retinas. The _green marker_ is the predicted probability value for the uploaded image. Generally, values below 50% are considered as belonging to the `Healthy` class, and values above 50% are classified as `Glaucoma`.")
        
        ## plot the position of this predicted probability value against the distribution of all probabilities generated from the test set
        
        # read .csv with test set probabilities         
        predictions_df = pd.read_csv('../data/test_predictions.csv')
        
        # create lists of glaucoma and healthy true labels from test set         
        glauc = predictions_df[predictions_df['label']=='Glaucoma']['preds']
        healthy = predictions_df[predictions_df['label']=='Healthy']['preds']
        var = [glauc, healthy]
        labels = ['Glaucoma', 'Healthy']
        
        # generate figure using true labels
        fig = ff.create_distplot(var, 
                                 labels, 
                                 show_hist=False, 
                                 show_rug=False)
        # set size and title
        fig.update_layout(width=600, 
                          height=400,
                          title_text='Predicted Probabilities for Pathological and Healthy Retinas',
                          title_y=0.92,
                          title_x=0.45,
                          xaxis_range=[-0.1, 1.1],
                          margin=dict(l=20, r=0, t=60, b=10),
                          paper_bgcolor="lightsteelblue",
                          legend=dict(font=dict(size=15)),
                          title_font=dict(size=18)
                         )
                         
        # set axes labels
        fig.update_xaxes(title='Predicted Probability', title_font_size=18, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Density in Distribution', title_font_size=18, tickfont_size=15, ticks='outside')
        
        # add interactive hover
        fig.update_traces(mode="lines", hovertemplate=None)
        fig.update_layout(hovermode="x unified")
        
        # add marker at position matching predicted probability of the uploaded image 
        fig.add_scatter(x=[round(prediction, 2)], 
                        y=[hp.find_y(round(prediction, 2),fig)], # use helper function 'find_y' to determine what the y coordinate for the marker should be
                        name='uploaded image', 
                        marker=dict(size=10, 
                                    line=dict(width=2),
                                    color='darkseagreen', 
                                    symbol="star-diamond"))
        
        # display plot
        st.plotly_chart(fig, use_container_width=False)