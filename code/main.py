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
        ('About Glaucoma',
         'Data Analysis and Modeling',
         'Make a prediction')
    )

# -------------------------------------------------------------------------------------------------------
# if page is 'Home' -------------------------------------------------------------------------------------    
if page=='About Glaucoma':
    
    with st.sidebar:
        st.markdown("[What is Glaucoma?](#about-glaucoma)", unsafe_allow_html=True)
        st.markdown("[Anatomy of the Eye](#anatomy-of-the-eye)", unsafe_allow_html=True)
        st.markdown("[The Optic Disc and Glaucoma](#the-optic-disc-and-glaucoma)", unsafe_allow_html=True)
        
    ## DESCRIPTION OF THE DISEASE
    
    st.header('What is Glaucoma?')
    st.write('''
    Glaucoma is a disease of the eye which causes the loss of nerve fibers, or _neuropathy_, that carry visual information from the eye to the brain causing a permanent, and irreversible loss of vision. The loss of nerve fibers occurs so slowly that it is generally asymptomatic until a significant amount of vision is lost. The only treatment currently available is to slow the progress of the disease further by reducing the pressure that the fluid naturally contained inside the eye exterts on the retina. This makes it critical that the disease is caught early and treatment begun before any loss of vision occurs.  
    ''')
    
    st.subheader('Anatomy of the Eye')
    col1, col2 = st.columns((4,2))
    
    with col1:
        st.image('https://cdn.britannica.com/78/4078-050-828D676A/section-eye.jpg', caption='Fig 1. Anatomy of the human eye.')
        st.image('../figures/retina_optic_nerve.jpg', caption='Fig 2. Cross-section of the retina, showing the optic nerve.')
    with col2:
        st.write('''
        The eye is a specialized organ that uses a complex network of cells that not only convert light into electricity, but also perform preliminary processing on them before transmitting the signals to the brain. This layer of cells lies at the back of the eye, as shown in `Fig 1.` ([_source_]('https://cdn.britannica.com/78/4078-050-828D676A/section-eye.jpg')), and is known as the _retina_. 
        
        The retina is a highly complex structure consisting of multiple layers of specialized cells (`Fig 2.` [_source_]('https://www.kenhub.com/en/library/anatomy/the-optic-nerve')). It is built somewhat backwards, with light having to pass through all the layers to impinge on the cells on the bottom-most layer which then convert light energy to electricity. This electrical signal then makes it way back towards the surface of the retina layer-by-layer, getting processed for visual information at each step. The fully process signal is then transmitted to the brain through the optic nerve. Since the optic nerve has to pass through the entire thickness of the retina to reach the brain, this section of the retina cannot process any light signals, and comprises the well-known _blind-spot_ of the human visual field. This region of the retina is also known as the _optic disc_.  
        ''')
    
    st.subheader('The Optic Disc and Glaucoma')
    st.write('''
    During an eye exam, the optic disc can be visualized clearly as a circular structure, with the nerve appearing as pinkish tissue. The shape and structure of the optic disc is used to gauge the health of the nerve tissue exiting the eye. A healthy optic disc consists of a broad rim which comprises the nerve fibers entering the optic nerve, and a small cup which is a shallow depression located centrally in the structure. As seen in `Fig 3.` ([source]('https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4523637/')), nerve loss causes the rim to become narrower and the cup to become deeper and wider.
    ''')
    
    col1, col2 = st.columns((2, 4))
    with col1:
        st.write('''
        This also results in a noticeable change in the appearance of the optic disc when visually examined using an opthalmoscope or photograph. `Fig 4.` ([source]('https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4523637/')) shows images of optic discs and the associated visual fields of three subjects with varying levels of nerve loss.  
        ''')
    with col2:
        st.image('../figures/anatomy.png', caption='Fig 4. Changes in optic disc anatomy due to Glaucomatous loss of nerve fibers.')
    
    st.image('../figures/glaucoma vs healthy photos.jpg', caption='Fig 4. Optic disc appearance and visual field loss in healthy vs glaucamatous retinas.', use_column_width='always')

# -------------------------------------------------------------------------------------------------------
# if page is 'Data Analysis and Modeling' -------------------------------------------------------------------------------------
if page == 'Data Analysis and Modeling':
    with st.sidebar:
        st.markdown("[Data](#data)", unsafe_allow_html=True)
        st.markdown("[Model Description](#model-description)", unsafe_allow_html=True)
        st.markdown("[Model Performance](#model-performance)", unsafe_allow_html=True)

    ## DATA -----------------------------------------------------------
    st.header('Data')
    st.subheader('''
    __Source__
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
    
    col1, col2, col3 = st.columns((3,0.5,3))
    
    with col1:
        avg = np.load('../data/avg_pixel_val_healthy.npy')
        fig = px.imshow(avg.reshape(178, 178),color_continuous_scale='viridis', aspect='equal', zmin=0.4, zmax=0.8)
        fig.update_layout(coloraxis=dict(colorbar=dict(len=0.5)),
                         width=400,
                         height=400,
                         title_text='Healthy Optic Disc',
                         title_y=0.9,
                         title_x=0.48)
        fig.update_xaxes(title='Pixel Position', title_font_size=15, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Pixel Position', title_font_size=15, tickfont_size=15, ticks='outside')
        
        # display plot
        st.plotly_chart(fig, use_container_width=False)
        st.caption('Average pixel value between 0 and 1 across 100 randomly sampled images of healthy optic discs.')
    with col3:
        avg = np.load('../data/avg_pixel_val_glaucoma.npy')
        fig = px.imshow(avg.reshape(178, 178),color_continuous_scale='viridis', aspect='equal', zmin=0.4, zmax=0.8)
        fig.update_layout(coloraxis=dict(colorbar=dict(len=0.5)),
                         width=400,
                         height=400,
                         title_text='Pathalogical Optic Disc',
                         title_y=0.9,
                         title_x=0.48)
        fig.update_xaxes(title='Pixel Position', title_font_size=15, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Pixel Position', title_font_size=15, tickfont_size=15, ticks='outside')
        # display plot
        st.plotly_chart(fig, use_container_width=False)
        st.caption('Average pixel value between 0 and 1 across 100 randomly sampled images of pathalogical optic discs.')
        
        
    
    ## MODEL DESCRIPTION -------------------------------------------------
    st.header('''
    __Model description:__
    ''')
    st.image('../figures/model_03_auc93.png', 
             caption='Fig 4. A schematic reprentation of the architecture of the CNN optimized for detecting Glaucoma in images of the optic nerve.',
            )
    
    ## MODEL PERFORMANCE ---------------------------------------------------
    st.header('''
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