#!/usr/bin/env python
# coding: utf-8
from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 99%; }
    div#menubar-container     { width: 99%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))
# # Import Visualization toos

# In[11]:


# Standard plotly imports
from pylab import rcParams
import matplotlib.pyplot as plt

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode, plot
import plotly.figure_factory as ff
import plotly.express as px
import plotly
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np


# #  Set Figure Properties_Matplotlib
# In[12]:
# get_ipython().run_line_magic('matplotlib', 'notebook')
''' figure properties'''
def fig_props(fig_size=(8.5,10),tick_sz = 15,legend_fontsz = 16,axes_labsz = 20):
    ''' figure params'''
    rcParams['font.family']='Times New Roman';
    rcParams['font.weight']='bold';
    rcParams["axes.labelweight"] = "bold"
    rcParams['figure.figsize'] = fig_size;
    rcParams['grid.linestyle']='-';
    rcParams['grid.color']='black';
    rcParams['legend.fontsize']=legend_fontsz;
    rcParams['axes.labelsize']=axes_labsz;
    rcParams['lines.linewidth']=2;
    plt.rc('xtick', labelsize=tick_sz)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=tick_sz)

fig_props((20,15))


# # Set Figure Properties_Plotly

# In[13]:
def templated_fig():
    ''' set default parameters for plotly '''
    pio.templates.default = "simple_white"
    fig1 = go.Figure(layout={
        'title': None,
        'font': {'size': 25, 'family': 'Rockwell', 'color': 'Black'},
        "title_font_family": "Times New Roman",
        "title_font_color": "red",
        "legend_title_font_color": "green",
        "xaxis": {"tickprefix":"<b>","ticksuffix" :"</b><br>","showgrid": True, "ticks": "inside", "tickson": "boundaries",
                  "linewidth":2,"ticklen": 8},
        "yaxis": {"tickprefix":"<b>","ticksuffix" :"</b><br>","showgrid": False, "ticks": "inside", "tickson": "boundaries",
                  "linewidth":2,},
        "scene":{'annotations':[dict(text='bold')]},
         },
            )#"height": 500, "width": 2300,
    return pio.to_templated(fig1)

# DataFrame Plot

def Plotly_plot(fig=None,df=None,subplot_titles=None,title=None,file='Test',ylab=None,xlab = 'Time',show_link=True,textfontsize=15,space_v=0.03,legend=True,tickformat= None,nticks_x=17,mode='lines',annot_font=16):
    colors = iter(px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet)
    marker_symbol = iter(['x','circle','star','square','diamond','x','circle','star','square','diamond'])
    if subplot_titles is None:
        subplot_titles=list(map(( lambda x: '<b>'+x+'</b>'), df.columns.str.replace('_',' ').tolist()));
    if fig==None:
        fig = make_subplots(rows=df.shape[1],cols=1,shared_xaxes=True,vertical_spacing=space_v,subplot_titles=subplot_titles)
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=annot_font)
    for i,col in  enumerate(df.columns):
        ylabel=ylab;
        if ylab!=None:
            ylabel='<b>'+ylab[i]+'</b>'

        fig.append_trace(go.Scatter(x=df.index, y=df[col].values,mode=mode,text=['Datapoint'],name=col,marker_color=next(colors),
                            showlegend=legend,marker_line_width=2, marker_size=5,marker_symbol=None),row=i+1,col=1)
        # This styles the line
        fig.update_traces(line=dict(width=3.0))
        fig.update_yaxes(title_text=ylabel, nticks=5,showline=True, linewidth=2,row=i+1, col=1)
        fig.update_xaxes(nticks=20,ticks="outside",zeroline=False,showline=True, linewidth=2, linecolor='black', row=i+1, col=1, range = [df.index[0],df.index[-1]],constrain="domain")
        fig.update_layout(template=templated_fig().layout.template, showlegend=legend)
        fig.update_layout(font=dict(family="Rockwell",size=textfontsize,color="black",),legend=dict(title=None, orientation="h", y=1.05, yanchor="bottom", x=0.75, xanchor="center"))
        fig.update_layout(height=None, width=None,title=title)
        fig.update_layout(hoverlabel=dict(bgcolor="lime",font_color="black",font_size=14,font_family="Rockwell"))
    fig.update_xaxes(title_text='<b>'+xlab+'</b>',nticks=nticks_x,ticks="outside",zeroline=False,showline=True, linewidth=2, linecolor='black', row=i+1, col=1, range = [df.index[0],df.index[-1]],constrain="domain")
    if tickformat:
        fig.update_xaxes(tickformat=tickformat, row=i+1, col=1)

    if show_link:
        plot(fig,show_link=show_link,filename=f'{file}.html')
    else:
        return fig


# # Decision boundary plot with Plotly

class visualize():
    def __init__(self,colorscale=px.colors.qualitative.Vivid[:2],colors=None,fig=None,marker_symbol=None):
        self.colorscale = colorscale
        if colors is None:
            self.colors = iter( px.colors.qualitative.Alphabet+px.colors.qualitative.Dark24)
        else:
            self.colors = iter(colors)
        pass

        if fig is None:
            self.fig = make_subplots(rows=1, cols=1,subplot_titles=None)
        else:
            self.fig = fig

        if marker_symbol is None:
            self.marker_symbol = iter(['x','circle','cross','star','square','plus','diagmond'])


    def plotly_decision_boundary(self,model=None,X=None,y=None,fig=None,h = .02,dim=2,row=1,col=1,X_test= None,y_test =None,Categories=[0,1],legend=True):
        X = np.array(X)
        y = np.array(y)
        if row==0 or col==0:
            raise('Row or Column Value should be grater than 0')
        if row+col>2:
            legend=False
          # print('legend disabled')
        if fig is None:
            fig = make_subplots(rows=1, cols=1,subplot_titles=None)

        ''' get min and max values '''
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        ''' create meshgrid of all the combinations'''
        plot_step=h;
        x_ = np.arange(x_min, x_max, plot_step)
        y_ = np.arange(y_min, y_max, plot_step)
        xx, yy = np.meshgrid(x_, y_)

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig.append_trace(go.Heatmap(x=x_, y=y_, z=Z,
                  colorscale=self.colorscale,
                  showscale=False,name='',showlegend=False),row=row,col=col)

        if (X_test is not None) and (y_test is not None):
            X = np.array(X_test);
            y = np.array(y_test)

        for i,label in enumerate(np.unique(y)):
            fig.append_trace(go.Scatter(x=X[y==label, 0], y=X[y==label, 1],mode='markers',text=['Datapoint'],name=Categories[i],marker_color=next(self.colors),
                            showlegend=legend,marker_line_width=1, marker_size=10,marker_symbol='circle'),row=row,col=col) # next(self.marker_symbol)

            fig.update_xaxes(nticks=0,ticks="",zeroline=True,showline=True,showticklabels=False, linecolor='black',range=[X[:, 0].min(),X[:, 0].max()],constrain="domain",row=row,col=col)
            fig.update_yaxes(nticks=0,ticks="",zeroline=True,showline=True,showticklabels=False, linecolor='black',range=[X[:, 1].min(),X[:, 1].max()],constrain="domain",row=row,col=col)
        fig.update_layout(template=templated_fig().layout.template)
        return fig
# In[ ]:

    ''' Needs proper implementation '''
    def mk_svm_decision_boundary(X,y):
        h = 0.2    
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1                                  #<------------- Change Features
        y_min, y_max = X[:, 4].min() - 1, X[:, 4].max() + 1                                  #<------------- Change Features
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))


            # title for the plots

        titles = 'MK-SVM Decision Boundary - RBF Kernels'

        supvec_p1 = [supvec1]


        dat = [np.expand_dims(np.c_[xx.ravel(), yy.ravel()][:,0],axis=1),np.expand_dims(np.c_[xx.ravel(), yy.ravel()]
                                                                                        [:,1],axis=1)]#np.c_[xx.ravel(), yy.ravel()]



        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.plot()



        Z = clf_op.predict(dat, feat_set = [0,1])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)


        # Plot also the training points and support vectors
        plt.scatter(X_train[:, 0], X_train[:, 4], c=y_train, cmap=plt.cm.coolwarm)   # <--------- Change Features
        plt.scatter(supvec_p1[0][:,0], supvec_p1[0][:,1], c="yellow", marker= '+', s=100)
        plt.xlabel('DT', fontsize=34)
        plt.ylabel('MC', fontsize=34)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.title(titles, fontsize=42)
        plt.show()

        plt.savefig('DB_linear.png')
