import streamlit as st
import streamlit.components.v1 as components
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import helper as hp
from pathlib import Path

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
         'Make a prediction',
         'References')
    )

# -------------------------------------------------------------------------------------------------------
# if page is 'About Glaucoma' -------------------------------------------------------------------------------------    
if page=='About Glaucoma':
    
    # provide hyperlinks to subheaders in sidebar for navigation:
    with st.sidebar:
        st.markdown("[What is Glaucoma?](#about-glaucoma)", unsafe_allow_html=True)
        st.markdown("[Anatomy of the Eye](#anatomy-of-the-eye)", unsafe_allow_html=True)
        st.markdown("[The Optic Disc and Glaucoma](#the-optic-disc-and-glaucoma)", unsafe_allow_html=True)
        st.markdown("[The need for affordable imaging and screening](#the-need-for-affordable-imaging-and-screening)", unsafe_allow_html=True)
        
    ## DESCRIPTION OF THE DISEASE
    
    # What is Glaucoma
    st.header('What is Glaucoma?')
    st.write('''
    Glaucoma is a disease of the eye which causes the loss of nerve fibers that carry visual information from the eye to the brain causing a permanent, and irreversible loss of vision. The loss of nerve fibers occurs so slowly that it is generally asymptomatic until a significant amount of vision is lost. The only treatment currently available is to slow the progress of the disease further by reducing the pressure that the fluid naturally contained inside the eye exterts on the retina. This makes it critical that the disease is caught early and treatment begun before any loss of vision occurs.  
    ''')
    
    #Anatomy of the eye
    st.subheader('Anatomy of the Eye')
    col1, col2 = st.columns((4,2))
    
    with col1:
        st.image('https://cdn.britannica.com/78/4078-050-828D676A/section-eye.jpg', caption='Fig 1. Anatomy of the human eye.')
        st.image('../figures/retina_optic_nerve.jpg', caption='Fig 2. Cross-section of the retina, showing the optic nerve.')
    with col2:
        st.write('''
        The eye is a specialized organ that uses a complex network of cells that not only convert light into electricity, but also perform preliminary processing on them before transmitting the signals to the brain. This layer of cells lies at the back of the eye, as shown in `Fig 1.` ([_source_](https://cdn.britannica.com/78/4078-050-828D676A/section-eye.jpg)), and is known as the _retina_. 
        
        The retina is a highly complex structure consisting of multiple layers of specialized cells (`Fig 2.` [_source_](https://www.kenhub.com/en/library/anatomy/the-optic-nerve)). It is built somewhat backwards, with light having to pass through all the layers to impinge on the cells on the bottom-most layer which then convert light energy to electricity. This electrical signal then makes it way back towards the surface of the retina layer-by-layer, getting processed for visual information at each step. The fully process signal is then transmitted to the brain through the optic nerve. Since the optic nerve has to pass through the entire thickness of the retina to reach the brain, this section of the retina cannot process any light signals, and comprises the well-known _blind-spot_ of the human visual field. This region of the retina is also known as the _optic disc_.  
        ''')
    
    # Optic Disc and Glaucoma
    st.subheader('The Optic Disc and Glaucoma')
    st.write('''
    During an eye exam, the optic disc can be visualized clearly as a circular structure, with the nerve appearing as pinkish tissue. The shape and structure of the optic disc is used to gauge the health of the nerve tissue exiting the eye. A healthy optic disc consists of a broad rim which comprises the nerve fibers entering the optic nerve, and a small cup which is a shallow depression located centrally in the structure. As seen in `Fig 3.` ([source](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4523637/)), nerve loss causes the rim to become narrower and the cup to become deeper and wider.
    ''')
    
    col1, col2 = st.columns((2, 4))
    with col1:
        st.write('''
        This also results in a noticeable change in the appearance of the optic disc when visually examined using an opthalmoscope or photograph. `Fig 4.` ([source](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4523637/)).  
        ''')
    with col2:
        st.image('../figures/anatomy.png', caption='Fig 3. Changes in optic disc anatomy due to Glaucomatous loss of nerve fibers.')
    
    st.image('https://www.glaucomaassociates.com/wp-content/uploads/2015/07/glaucoma-cupping.jpg', caption='Fig 4. Optic disc appearance and visual field loss in healthy vs glaucamatous retinas.', use_column_width='always')
    
    # The need for affordable imaging and screening
    st.subheader('The need for affordable imaging and screening')
    st.write('''
    Glaucoma progresses very slowly, taking 10-15 years, on average, to achieve total blindness. As a result, it is asymptomatic in the early stages and cango unnoticed for years. Patients who seek out medical care for glaucoma on their own without any prior screening only do so after noticable vision loss has already occured. This makes it critical to screen for glaucoma as part of a standard eye exam. Currently screening is done by a trained professional who visually examines the inside of the eye using a photograph or opthalmoscope to identify any abnormal anatomical structures, including nerve fiber loss due to glaucoma. 
    
    However, access to a trained professional, or expensive machines to take photographs of the internal structures of the eye is limited in rural or underserved areas, especially in poorer parts of the world. A combination of an inexpensive imaging device and a reliable predictive model dispersed across these areas with a technician trained to use them would hugely benefit communities and provide necessary preventative care to slow down the progress of the disease and preserve vision.  
    
    Extensive research in these areas has already produced means to take photographs of the internal eye using a smart phone combined with lenses. (`Fig 5.`, `Fig 6.`, `Fig 7.` [source](https://advanceseng.com/miniaturized-indirect-ophthalmoscopy-foster-smartphone-wide-field-fundus-photography/), [publication](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4951186/))
    ''')
    col1, col2, col3 = st.columns((3, 0.5, 3))
    with col1:
        st.image('../figures/retina_smart_phone.jpg', caption='Fig 5. (a) Smartphone with adaptor for fundus photography. (b) Photograph of separate components of the adaptor. (caption from image source)', width = 300)
    with col3: 
        st.image('../figures/retina_smart_phone_ex.jpg', caption='Fig 6. (a) Optical layout of the wide-field fundus imaging device. (b) Photographic illustration of trans-palpebral illumination. (caption from image source)', width = 340)
    
    col1, col2, col3 = st.columns((0.5, 4, 0.5))
    with col2:
        st.image('../figures/retina_imaging.jpg', caption='Fig 7. Example images taking using the trans-palpebral illumination technique with a smart phone.')
    
    st.write('''
    Additionally, studies also exist that have focused their research on automating the detection of glaucoma pathology in images of the retina. [Diaz-Pinto et.al](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-019-0649-y#Sec2), in particular, use a convolutional neural network to assess images for glaucomatous loss of nerve tissue. 
    
    In the following pages, I have described a simple convolutional neural network that I trained on the same dataset as [Diaz-Pinto et.al](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-019-0649-y#Sec2). A comparison of the performance metrics of my CNN and the the Diaz-Pinto model are presented below:
    ''')
    col1, col2, col3 = st.columns((1, 2, 1))
    with col2:
        metrics = pd.read_csv('../data/metrics.csv')
        metrics = metrics.drop(columns='train set')
        metrics = metrics.rename(columns={'test set' : 'current model'})
        metrics.index = ['recall', 'accuracy', 'auc', 'precision']
        st.table(metrics)

        
# ----------------------------------------------------------------------------------------------------------------------------
# if page is 'Data Analysis and Modeling' -------------------------------------------------------------------------------------
if page == 'Data Analysis and Modeling':
    
    # provide hyperlinks to subheaders in sidebar for navigation:
    with st.sidebar:
        st.markdown("[Data](#data)", unsafe_allow_html=True)
        col1, col2 = st.columns((0.1,2))
        with col2:
            st.markdown("[Source](#source)", unsafe_allow_html=True)
            st.markdown("[Exploratory Data Analysis](#exploratory-data-analysis)")
        st.markdown("[Convolution Neural Network](#convolution-neural-network)", unsafe_allow_html=True)
        col1, col2 = st.columns((0.1,2))
        with col2:
            st.markdown("[Model Performance](#model-output-and-threshold-optimization)", unsafe_allow_html=True)

    ## DATA -----------------------------------------------------------
    st.header('Data')
    st.subheader('__Source__')
    st.write('''
    I obtained 705 publicly avialable fundus images from the [ACRIMA database](https://figshare.com/s/c2d31f850af14c5b5232). They were collected at the [FISABIO Oftalmología Médica](http://fisabio.san.gva.es/fisabio-oftamologia) in Valencia, Spain. 396 of these images are of glaucomatous optic discs and 309 are healthy. The images were obtained from consenting subjects, and labeled as glaucomatous or healthy by expert Glaucoma specialists. All images have been cropped around the optic disc to place it in a central position. `Fig 1.` shows a sample of five glaucomatous, and five healthy optic discs. 
    ''')
    
    # display images
    
    glaucoma_images = pd.read_csv('../data/glaucoma_images.csv')
    healthy_images = pd.read_csv('../data/healthy_images.csv')
    
    fig, ax = plt.subplots(2, 5, figsize=(25,10))
    fig.suptitle('Images of Glaucomatous (top row) and Healthy (bottom row) Optic Disks', 
                 size=30)

    plt.rcParams.update({'font.sans-serif':'Calibri'})

    plt.subplots_adjust(wspace=0.1,
                       hspace=0.05)

    for i, f in enumerate(glaucoma_images['file_path']):
        ax[0,i].imshow(Image.open(f))
        ax[0,i].set_title(r"$\bf{Glaucoma: }$" + f"{Path(f).name[0:5]}", size=18)
        ax[0,i].tick_params(
            axis='both',      
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False) 

    for i, f in enumerate(healthy_images['file_path']):
        ax[1,i].imshow(Image.open(f))
        ax[1,i].set_title(r"$\bf{Healthy: }$" + f"{Path(f).name[0:5]}", size=18)
        ax[1,i].tick_params(
            axis='both',      
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False) 
   
    # display figure
    st.pyplot(fig)
    st.caption('Fig 1. A randomly selected sample of five glaucomatous (top row), and healthy (bottom row) optic discs from the ACRIMA database.')
    
    st.subheader('Exploratory Data Analysis')
    st.write('''
    All images were color images with R, G and B channels. I first examined the distribution of pixel values across both classes to see if there were any differenes or uniqueness in the pixel values. `Fig 2. (Left)` Shows the distrubutions of the pixel values from the raw images ranging between 0 and 255. `Fig 2. (Right)` shows the distribution of pixel values normalized between 0 and 1, where 1 is the brightest spot in the image and 0 is the darkest. This was done to make the images more comparable since they varied in lighting and contrast. 
    ''')
    
    col1, col2, col3 = st.columns((3,0.25,3))
    
    with col1:
        pixels_dist_raw = pd.read_csv('../data/pixels_dist_raw.csv')

        fig = px.line(data_frame=pixels_dist_raw, x='Density', y='value', color='variable', color_discrete_sequence=['#1f77b4', '#ff7f0e'])

        # Add title
        fig.update_layout(width=400,
                          height=350,
                          title_text='Distribution of Pixel Values (Raw Image)',
                          title_y=0.95,
                          title_x=0.45,
                          plot_bgcolor='white',
                          template='plotly_white',
                          showlegend=False,
                          title_font=dict(size=15))

        fig.update_xaxes(title='Pixel Value (0 to 255)', title_font_size=15, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Density', title_font_size=15, tickfont_size=15, ticks='outside')

        st.plotly_chart(fig, use_column_width=True)
    
    with col3:
        pixels_dist_normalized = pd.read_csv('../data/pixels_distribution_normalized.csv')

        fig = px.line(data_frame=pixels_dist_normalized, x='Density', y='value', color='variable', color_discrete_sequence=['#1f77b4', '#ff7f0e'])

        # Add title
        fig.update_layout(width=400,
                          height=350,
                          title_text='Distribution of Pixel Values (Normalized Image)',
                          title_y=0.95,
                          title_x=0.45,
                          plot_bgcolor='white',
                          template='plotly_white',
                          legend=dict(font=dict(size=15)),
                          title_font=dict(size=15))

        fig.update_xaxes(title='Pixel Value (0 to 1)', title_font_size=15, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Density', title_font_size=15, tickfont_size=15, ticks='outside')

        st.plotly_chart(fig, use_column_width=True)
        
    st.caption('Fig 2. (__Left__) Distribution of raw pixel values (0 to 255) across all three channels (RGB) for all images in the database. (__Right__) Distribution of pixel values normalized between 0 (darkest pixel in the image) to 1 (brightest pixel in the image) for every channel (RGB) of every image in the database.')
    
    st.write('''
    I also looked at pixel value for the normalized images at each pixel position averaged across the three color channels look for any noticeable differences between glaucomatous and healthy optic discs (`Fig 3`)
    ''')
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
        
    with col3:
        avg = np.load('../data/avg_pixel_val_glaucoma.npy')
        fig = px.imshow(avg.reshape(178, 178),color_continuous_scale='viridis', aspect='equal', zmin=0.4, zmax=0.8)
        fig.update_layout(coloraxis=dict(colorbar=dict(len=0.5)),
                         width=400,
                         height=400,
                         title_text='Glaucomatous Optic Disc',
                         title_y=0.9,
                         title_x=0.48)
        fig.update_xaxes(title='Pixel Position', title_font_size=15, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Pixel Position', title_font_size=15, tickfont_size=15, ticks='outside')
        # display plot
        st.plotly_chart(fig, use_container_width=False)
        
    st.caption('Fig 3. (__Left__) Average pixel value between 0 and 1 across 100 randomly sampled images of healthy optic discs. (__Right__) Average pixel value between 0 and 1 across 100 randomly sampled images of pathalogical optic discs.')
        
    col1, col2, col3 = st.columns((3,1,3))
        
    with col1: 
        sizes_df = pd.read_csv('../data/image_sizes.csv')
        hist_data = [sizes_df[sizes_df['label']=='Healthy']['h'].values, sizes_df[sizes_df['label']=='Glaucoma']['h'].values]
        group_labels = ['Healthy', 'Glaucoma']
        colors = ['#1f77b4', '#ff7f0e']
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)
        fig.update_layout(width=400,
                          height=350,
                          title_text='Distribution of Image Heights for Raw Images',
                          title_y=0.8,
                          title_x=0.45,
                          plot_bgcolor='white',
                          template='plotly_white',
                          legend=dict(font=dict(size=15)),
                          title_font=dict(size=15))

        fig.update_xaxes(title='Image Height (in Pixels)', title_font_size=15, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Density', title_font_size=15, tickfont_size=15, ticks='outside')
        st.plotly_chart(fig, use_container_width=False)
        st.caption('Fig 4. Distribution of image heights in pixels for glaucomatous and healthy images.')
        
    with col3:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write('''
        Further, the raw images were varied in size. Since convolutional neural networks require all observations to have the same dimensions, I resized all images to the same size of (178 by 178) pixels, which was the size of the smallest image in the database.  
        ''')
    
    ## MODEL DESCRIPTION -------------------------------------------------
    st.header('__Convolution Neural Network__')
    st.markdown(
    '''
    A Convolutional Neural Network is the most commonly used deep learning model for classification or analyses of images. The network I used is a __Tensorflow Keras Sequential__ model with the following architecture:  
    
    Layer | Type | Hyperparameters |
    ------|------|-------------|
    1| Convolution (Conv2D) | `filters`=32, `kernel_size`=(4, 4), `activation`='relu', `input_shape`=(178, 178, 3)
    2| Pooling (MaxPooling2D)| `pool_size`=(2,2)
    3| Convolution (Conv2D) | `filters`=32, `kernel_size`=(4, 4), `activation`='relu', `input_shape`=(178, 178, 3)
    4| Pooling (MaxPooling2D)| `pool_size`=(2,2)
    5| Flattening (Flatten)| n/a
    6| Dense layer (Dense) | `units`=128, `activation`='relu'
    7| Output layer (Dense)| `units`=1, `activation`='sigmoid'
    8| Early stop (EarlyStopping)| `patience`=5
    
    The model was compiled with a `batch_size` of 256 and trained for 20 `epochs`. The images that the model was trained on were the raw images resized to 178 x 178 pixels. Glaucomatous images were labelled `1` and healthy images were labelled `0`. 
    
    A 2D schematic representation of the architecture of the model is presented below in `Fig 5.`, which was created using the [VisualKeras](https://github.com/paulgavrikov/visualkeras) library.    
    
    '''
    )
    st.image('../figures/model_03_auc93.png', 
             caption='Fig 5. A schematic representation of the architecture of the CNN optimized for detecting Glaucoma in images of the optic disc.',
            )
    
    ## MODEL PERFORMANCE ---------------------------------------------------
    st.subheader('Model Output and Threshold Optimization')
    st.markdown('''
    The metric chosen for optimization was recall, or sensitivity. Recall is a metric that measure how good a model is at detecting observations that truely belong to the positive class. Therefore, optimizing for recall minimizes the number of false negatives, which in this case is classifying a glaucomatous image as heathy.  
    
    The output of the CNN is a prediction ranging between 0 and 1. The value represents the probability of the input image belonging to the positive class, which in this case, is **Glaucoma**. A distribution of probability values that were predicted for healthy and glaucomatous images in the test set, which the model had no prior exposure to, are shown in `Fig 6.`.  
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
                                      font=dict(color='black'),
                                      x=0.7, y=0.95, 
                                      bordercolor='olivedrab'))
        # display plot
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Fig 6. Distributions of predicted probability for healthy and galucomatous images in the test set.')
        
    with col3:
        st.markdown('''
        It is clear from the figure that the two distributions are very clearly separated with a wide range between the two peaks which could serve as a threshold value of probability to classify any given image into the **`Healthy`** or **`Glaucoma`** class.  
        
        To determine the most optimal threshold value, I calculated performance metrics at 20 values between 0.3 and 0.8. The metric scores are shown in `Fig 7.`  
        ''')
    
    st.markdown('''
       
        The metrics calculated were:
        
        Metric | Interpretation
        -------|---------------
        Recall | Higher values minimize the number of instances of glaucomatous images being classified as healthy
        Precision| Higher values minimize the number of healthy images being classified as glaucomatous
        Accuracy | The overall percentage of correct classifications. This is still relevant since both classes had a comparable number of observations.
        AUC | Higher values implies better classification overall. 
        
        ''')
    st.write('')            
    '''        
    Metric scores at different scores is shown below in `Fig 7.`
    '''
    
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
    fig.update_xaxes(title='Threshold value for Predicted Probability', title_font_size=15, tickfont_size=15, ticks='outside')
    fig.update_yaxes(title='Metric Value', title_font_size=15, tickfont_size=15, ticks='outside')

    fig.update_traces(mode="lines", hovertemplate=None)
    fig.update_layout(hovermode="x")

    # add retangles
    fig.add_vrect(x0=0.3, x1=0.589, line_width=0, fillcolor="palegoldenrod", opacity=0.2, annotation=dict(text='best recall', x=0.58, y=0.96, bordercolor='goldenrod', font=dict(color='black')))
    fig.add_vrect(x0=0.589, x1=0.615, line_width=0, fillcolor="lightslategray", opacity=0.2, annotation=dict(text='best auc', x=0.63, y=0.96, bordercolor='slategrey', font=dict(color='black')))

    # display plot
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption('Fig 7. Metric scores as a function of threshold value. The pale yellow shaded region represents the threshold values that have the best recall score before it begins to show a steep decline. The grey shaded area represents the range of threshold values where the area under the curve for the ROC of the classification was the best. ')
    
    col1, col2, col3 = st.columns((3,0.5,2))
    
    with col1:

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
        st.caption('Fig 8. Confusion matrix for the classification of images in the test set with predicted class along the x-axis and true class on the y-axis.')
    
    with col3:
        '''
        While a recall score of `1` was possible to achieve at low threshold values of about `0.3`, this resulted in a large number of healthy images being classified as glaucomatous. A threshold close to `0.6` had the highest accuracy and AUC scores, while maintaining a recall of above `0.95`. I therefore selected `0.589` which was the threshold at which recall was the highest within the grey region. This yielded the confusion matrix in `Fig 8.` and the metric scores shown below: 
        '''     
    col1, col2, col3 = st.columns((2,3,2))
    
    with col2:
        metrics = pd.read_csv('../data/metrics.csv')
        metrics = metrics.drop(columns=['Diaz-Pinto Model'])
        metrics.index = ['recall', 'accuracy', 'auc', 'precision']
        st.table(metrics)
        

# --------------------------------------------------------------------------------------------------
# if page is 'Make a Prediction' -------------------------------------------------------------------    
if page == 'Make a prediction':
    # set up file uploader 
    st.subheader('Use this page to interact with the trained model. Upload an image of the retina using the box below: ')
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
        
        st.subheader(f"The probability that this retina is Glaucomatous is __{round(prediction*100, 2)}%__.")
        st.write("The image below shows the distributions of predicted probability of belonging to the class 'Glaucoma' for images of healthy (_orange_) and pathological (_blue_) retinas. The _green marker_ is the predicted probability value for the uploaded image. The model classifies images with probabilities below 58.9% as `Healthy`, and those above 58.9% as `Glaucoma`.")
        
        ## plot the position of this predicted probability value against the distribution of all probabilities generated from the test set
        
        predictions_df = pd.read_csv('../data/test_predictions.csv')

        hist_data = [predictions_df[predictions_df['label']=='Healthy']['preds'], predictions_df[predictions_df['label']=='Glaucoma']['preds']]
        group_labels = ['Healthy', 'Glaucoma']
        colors = ['#1f77b4', '#ff7f0e']

        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)

        # Add title
        fig.update_layout(width=500,
                          height=500,
                          title_text='Distribution of Predicted Probabilities',
                          title_y=0.9,
                          title_x=0.45,
                          plot_bgcolor='white',
                          template='plotly_white',
                          legend=dict(font=dict(size=15)),
                          title_font=dict(size=18))

        fig.update_xaxes(title='Predicted Probability', title_font_size=15, tickfont_size=15, ticks='outside')
        fig.update_yaxes(title='Density', title_font_size=15, tickfont_size=15, ticks='outside')
        
        # add interactive hover
        fig.update_traces(mode="lines", hovertemplate=None)
        fig.update_layout(hovermode="x unified")
        
        fig.add_vrect(x0=0, x1=0.589, 
                      line_width=0, 
                      fillcolor="#1f77b4", 
                      opacity=0.2, 
                      annotation=dict(text='Healthy',
                                      font=dict(color='black'),
                                      x=0.3, y=0.95, 
                                      bordercolor='#1f77b4'))
        
        fig.add_vrect(x0=0.589, x1=1, 
                      line_width=0, 
                      fillcolor="#FFC999", 
                      opacity=0.2, 
                      annotation=dict(text='Glaucoma',
                                      font=dict(color='black'),
                                      x=0.9, y=0.95, 
                                      bordercolor='#FFC999'))
        
        # add vertical line to indicate the location of the uploaded image
        fig.add_vline(x=round(prediction, 2), 
                      line_width=2, 
                      line_dash="dash", 
                      line_color="red",
                      annotation=dict(text="uploaded image",
                                     font=dict(color='black'),
                                     font_size=15,
                                     textangle=-90),
                     annotation_position='bottom right')
       
        # display plot
        st.plotly_chart(fig, use_container_width=False)
        st.caption('Distributions of predicted probability for healthy and galucomatous images in the test set.')
        
        
        
        
        
        
        
#         # read .csv with test set probabilities         
#         predictions_df = pd.read_csv('../data/test_predictions.csv')
        
#         # create lists of glaucoma and healthy true labels from test set         
#         glauc = predictions_df[predictions_df['label']=='Glaucoma']['preds']
#         healthy = predictions_df[predictions_df['label']=='Healthy']['preds']
#         var = [glauc, healthy]
#         labels = ['Glaucoma', 'Healthy']
        
#         # generate figure using true labels
#         fig = ff.create_distplot(var, 
#                                  labels, 
#                                  show_hist=False, 
#                                  show_rug=False)
#         # set size and title
#         fig.update_layout(width=600, 
#                           height=400,
#                           title_text='Predicted Probabilities for Glaucomatous and Healthy Retinas',
#                           title_y=0.92,
#                           title_x=0.45,
#                           xaxis_range=[-0.1, 1.1],
#                           template='plotly_white',
#                           legend=dict(font=dict(size=15)),
#                           title_font=dict(size=18)
#                          )
                         
#         # set axes labels
#         fig.update_xaxes(title='Predicted Probability', title_font_size=18, tickfont_size=15, ticks='outside')
#         fig.update_yaxes(title='Density in Distribution', title_font_size=18, tickfont_size=15, ticks='outside')
        
#         # add interactive hover
#         fig.update_traces(mode="lines", hovertemplate=None)
#         fig.update_layout(hovermode="x unified")
        
#         # add marker at position matching predicted probability of the uploaded image 
#         fig.add_scatter(x=[round(prediction, 2)], 
#                         y=[hp.find_y(round(prediction, 2),fig)], # use helper function 'find_y' to determine what the y coordinate for the marker should be
#                         name='uploaded image', 
#                         marker=dict(size=10, 
#                                     line=dict(width=2),
#                                     color='darkseagreen', 
#                                     symbol="star-diamond"))
        
#         # display plot
#         st.plotly_chart(fig, use_container_width=False)

# -------------------------------------------------------------------------------------------------------
# if page is 'About Glaucoma' -------------------------------------------------------------------------------------    
if page=='References':
    st.header('References')
    st.write('__Anatomy of the Human Eye__')
    col1, col2 = st.columns((0.2,5))
    with col2:
        st.caption('[Encyclopedia Britannica](https://www.britannica.com/science/human-eye)')
        st.caption('[Ken Hub, Optic Nerve](https://www.kenhub.com/en/library/anatomy/the-optic-nerve)')
    
    st.write('__Pathophysiology of Glaucoma__')
    col1, col2 = st.columns((0.2,5))
    with col2:
        st.caption('Review paper by [Weinreb et.al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4523637)')
    
    st.write('__Affordable Solution For Fundus Imaging__')
    col1, col2 = st.columns((0.2,5))
    with col2:
        st.caption('Publication by [Toslak et.al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4951186/)')
        st.caption('Advances in Engineering [Article](https://advanceseng.com/miniaturized-indirect-ophthalmoscopy-foster-smartphone-wide-field-fundus-photography/)')
    
    st.write('CNNs for Automated Glaucoma Assessment')
    col1, col2 = st.columns((0.2,5))
    with col2:
        st.caption('Publication by [Diaz-Pinto et. al](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-019-0649-y#Sec2)')
        
    st.write('CNN Visualization')
    col1, col2 = st.columns((0.2,5))
    with col2:
        st.caption('[VisualKeras](https://github.com/paulgavrikov/visualkeras) library by Paul Gavrikov.')
    
  