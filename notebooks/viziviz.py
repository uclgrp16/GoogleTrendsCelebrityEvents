import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.formula.api import ols
import country_converter as coco
import plotly.graph_objects as go
#from google.colab import drive
import ruptures as rpt
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta
import os

class Visualizer:
    def __init__(self, keyword, event_date, event_info):
        self.keyword = keyword
        self.event_date = pd.to_datetime(event_date)
        self.event_info = event_info
        self.dict_item = self.gen_axis_dict()
    
    def gen_axis_dict(self):
        dict_item = {}
        dict_item['time_numeric'] = 'Time numeric'
        dict_item['harmonized_daily'] = 'Harmonized Daily'
        dict_item['harmonized_daily_rescaled'] = 'Rescaled Harmonized Daily'
        dict_item['raw_daily'] = 'Raw Daily'
        dict_item['date'] = 'Date'
        dict_item['dg'] = 'Smoothed Data (Polynomial Fit)'
        dict_item['region'] = 'Region'
        dict_item['brand'] = 'Brand'
        dict_item['month'] = 'Month'
        dict_item['day'] = 'Day of Month'
        dict_item['year'] = 'Year'
        dict_item['days'] = 'Day of year'
        dict_item['days_since_event'] = 'Days Elapsed from Event'
        self.default = f'Daily Searches for "{self.keyword}"'
        return dict_item


    def poly_eval(self, df, x_name, y_name, degree):
            """Creates a column with the interpolated polynomial with degree (for curvy lines)
                Args:
                df: dataframe
                x_name (str): the x variable (usually date or time_numeric) --> column in df
                y_name (str): the y variable (Google Trends data) --> column in df
                degree (int): degree to use
                """
            df['dg'] = np.polyval(np.polyfit(df[x_name].index, df[y_name], degree), df[x_name].index)
            return df

    # Data Frames Loaders
    def load_df(self, csv_path,index_col=None):
        """Loads a csv file as DataFrame and applies the necessary changes"""
        df = pd.read_csv(csv_path,index_col=index_col)
        try:
            df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
        except:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        event_numeric = pd.to_datetime(self.event_date).day_of_year
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['days'] = df['date'].dt.day_of_year
        df['time_numeric'] = df['date'].dt.day_of_year
        df['days_since_event'] = df['days'] - event_numeric
        return df
    
    def load_geo(self, input_dir):
        """Reads all files in a directory for the "Interest by Regions" section of Google Trends """
        dfs = []
        for file in os.listdir(input_dir):
            df = pd.read_csv(os.path.join(input_dir,file),skiprows=int(2))
            col = df.columns[int(1)]
            time_range = pd.to_datetime(col.split(': ')[-1].split(' ')[0][1:])
            df['date'] = time_range
            df.rename(columns={col:'raw_daily'},inplace=True)
            dfs.append(df)
        df_geo = pd.concat(dfs,ignore_index=True).dropna()
        df_geo['raw_daily'] = df_geo['raw_daily'].astype(str)
        df_geo['raw_daily'] = df_geo['raw_daily'].apply(lambda x: float(x.replace('<1','0')))
        return df_geo
    

    def remove_leap_days(self, df):
        """Removes leap days from the dataframe"""
        df = df[~((df.year % 4 == 0) & (df.month == 2) & (df.day == 29))]
        return df

    ## Basic Filterings 
    def filter_by_date(self, df,filter):
        """Filter the DataFrame for a specfic daterange"""
        filter1 = self.event_date + timedelta(int(filter))
        filter2 = self.event_date - timedelta(int(filter))
        filtered_df = df[(df['date'] <= filter1) & (df['date'] >= filter2)]
        return filtered_df.reset_index(drop=True)

    def filter_by_days(self,df,filter,column='time_numeric'):
        df = df[(df[column] <= (int(filter)))&(df[column] >= (-int(filter)))]
        return df.reset_index(drop=True)
    
    #filter for before/after event
    def get_before_after(self, df):
        if 'post_event' in df.columns:
            before_event = df[df['post_event'] == 0]
            after_event = df[df['post_event'] ==  1]
        elif 'post' in df.columns:
            before_event = df[df['post'] == 0]
            after_event = df[df['post'] ==  1]
        else:
            before_event = df[df['date'] < self.event_date]
            after_event =  df[df['date'] >= self.event_date]
        return before_event, after_event
    

    #Basic Visualisations

    ## Line Graph with Plotly
    def px_line_graph(self,df, x_name,y_name,event_date=None,z_name=None,title=None):
        """Generates a line graph for y over x"""
        event_date = self.event_date if not event_date else event_date
        if not z_name:
            fig = px.line(df, x=x_name, y=y_name, title=title)
        else:
            fig = px.line(df, x=x_name, y=y_name,color=z_name, title=title)
        fig.add_shape( go.layout.Shape(type="line", x0=event_date, x1=event_date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="black", dash="dash")))
        if y_name != 'dg':
            fig.update_yaxes(range=[0, 110])
        fig.update_yaxes(title=self.dict_item[y_name])
        fig.update_xaxes(title=self.dict_item[x_name])
        return fig
    
    ## Polynomial Line Plot
    def poly_plot(self,df,x_name,y_name,degree,title=None):
        """Generates a polynomial fitted line for chosen degree"""
        df = self.poly_eval(df, x_name,y_name, degree)
        if not title:
            title = f'Polynomial Fit of {df.region.unique()[0]} {self.default}'
        return self.px_line_graph(df, x_name,'dg',title=title)

    def plot_multi_lines(self, df, x_name, y_name1, y_name2,title=None):
        """Generates a plot for two lines"""
        if not title:
            title = f'{self.default} - {self.dict_item[y_name1]} vs {self.dict_item[y_name2]}'
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[x_name], y=df[y_name1], line=dict(color='red'), mode='lines',name=self.dict_item[y_name1]))
        fig.add_trace(go.Scatter(x=df[x_name], y=df[y_name2], line=dict(color='blue'), mode='lines',name=self.dict_item[y_name2]))
        fig.update_layout( title=title, xaxis_title=self.dict_item[x_name], yaxis_title='Search Volume')
        return fig


    ## Box Plot
    def box_plot(self, df,y_name):
        """Makes a box plot comparing before/after event"""
        labels=['Before Event', 'After Event']
        sns.boxplot(x='post_event', y=y_name, data=df,palette=['green','blue'])
        xlabel = f'After Cutoff Date - {self.event_info}'
        plt.xlabel(xlabel)
        ylabel = self.dict_item[y_name]
        plt.ylabel(self.dict_item[y_name])
        plt.xticks(ticks=[0,1], labels=labels)
        title = f'{ylabel} with {xlabel}'
        plt.title(title)
        plt.show()
    
    ## Heatmap 
    def heatmap_category(self, df,x_name, y_name, z_name):
        """"Creates a Heatmap by Region"""
        heatmap_data = pd.pivot_table(df, values=y_name, index=x_name, columns=z_name, aggfunc=np.mean)
        fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns,y=heatmap_data.index, colorscale='thermal'))
        fig.update_layout( title=f'Heatmap for {self.default} by Region', xaxis_title='Region', yaxis_title='Date', width=800, height=600)
        return fig

    ##Bar Char
    def px_bar_graph(self,df, x_name,y_name,title=None):
        if not title:
            region = df['regions'].unique()[0]
            title = f'{region} Daily Searches for "{self.keyword}"'
        fig = px.bar(df, x=x_name, y=y_name, color=y_name,title=title)
        fig.update_yaxes(title=self.dict_item[y_name])
        fig.update_xaxes(title=self.dict_item[x_name])
        return fig
    
    def bar_char_by_categories(self, df,x_name, y_name,color_name,z_name,title=None):
        """Args:
            df - DataFrame
            x_name - the x variable
            y_name - the y variable
            color_name - which column to use data for color
            z_name - the facet_col """
        if not title:
            title = f'{x_name.capitlialize()} Searches for "{self.keyword}" by {z_name}'
        fig = px.bar(df, x=x_name, y=y_name, color=color_name, facet_col=z_name, title=title)
        #fig.update_yaxes(title='Harmonized Daily')
        fig.update_xaxes(title=x_name.capitalize())
        return fig 


    # Figure Constructors - these take an input Figure and adds the relevant lines if needed

    # Make the background white for the figure
    def make_white_background(self,fig):
        """Makes White Background for Figure"""
        fig.update_layout(plot_bgcolor='white' )
        fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
        return fig

    ## add 'linear' line - line of best fit
    def gen_linear(self, fig, df,x_name, y_name, id):
        linear_df =  self.poly_eval(df,x_name,y_name,1)
        fig.add_trace(go.Scatter(x=linear_df[x_name], y=linear_df['dg'], line=dict(color=px.colors.qualitative.Set2[id],dash='dash'), mode='lines'))
        return fig
    
    ## add 'polynomial fit' line with chosen degree, id being the color to use
    def gen_poly(self, fig, df, x_name, y_name,degree,id):
        linear_df =  self.poly_eval(df,x_name,y_name,degree)
        fig.add_trace(go.Scatter(x=linear_df[x_name], y=linear_df['dg'], line=dict(color=px.colors.qualitative.Dark2[id]), mode='lines'))
        return fig

   ## add scatter plot for chosen 
    def gen_scatter(self, fig, df, x_name, y_name,id, size=6):
        scatter_trace = px.scatter(df, x=x_name, y=y_name, color_discrete_sequence=[px.colors.qualitative.Pastel2[-(id)]])
        scatter_trace.data[0].marker.size = size
        fig.add_trace(scatter_trace.data[0])
        return fig
    
    ## Add Vertical Line denoting the line for our specific event date
    def add_event_line(self, fig, df, x_name):
        x_col = df[df['date'] == self.event_date].iloc[int(0)][x_name]
        fig.add_shape(go.layout.Shape(type="line",x0=x_col, x1=x_col, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash")))
        return fig

    ### Add vertical line, except we chose the value
    def add_event_line_chosen(self, fig, value):
        fig.add_shape(go.layout.Shape(type="line",x0=value, x1=value, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", dash="dash")))
        return fig

    # Models
    # 1. Seasonality
    ### Plot Seasonal Data
    def plot_seasonal_data(self,df, y_name,period):
        seasonal_decomposed = seasonal_decompose(df[y_name], model='additive', period=period)
        region = df.region.unique()[0]
        y_original = df[y_name].values
        x_original = df.index
        fig = make_subplots(rows=int(2), cols=int(2), subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
        y_trend = seasonal_decomposed.trend.values
        y_seasonal = seasonal_decomposed.seasonal.values
        y_residual =  seasonal_decomposed.resid.values
        fig.add_trace(go.Scatter(x=x_original, y=y_original, mode='lines', name='Original'), row=int(  1), col=int(1))
        fig.add_trace(go.Scatter(x=x_original, y=y_trend, mode='lines', name='Trend'), row=int(1), col=int(2))
        fig.add_trace(go.Scatter(x=x_original, y=y_seasonal, mode='lines', name='Seasonal'), row=int(2), col=int(1))
        fig.add_trace(go.Scatter(x=x_original, y=y_residual, mode='lines', name='Residual'), row=int(2), col=int(2))
        fig.update_layout(title=f'{region} - "{self.keyword}" Search Volume Decomposition')
        return fig
    
    #2. PELT

    ## Get Change Points
    def get_change_points(self, df, x_name, y_name, pen_param=100):
        algo = rpt.Pelt(model="l2", jump=int(1), min_size=int(1)).fit(df[[y_name]])
        change_points = algo.predict(pen=int(pen_param))
        change_points2dates = [df.iloc[x][x_name] for x in change_points[:-1]]
        if x_name == 'date':
            return change_points2dates
        else:
            return change_points

    ### Plot Pelt Model l2 for single region
    def pelt_model_l2(self, df, x_name,y_name,pen_param=100):
        results =  self.get_change_points(df, x_name, y_name, pen_param)
        fig = go.Figure()
        y_values = list(df[y_name])
        x_values = list(df[x_name])
        region_name = df['region'].unique()[0]
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Search Volume'))
        ##removing the last result because it's out of bounds with the index
        fig.add_trace(go.Scatter(x=results, y=[y_values[x_values.index(cp)] for cp in results[:-1]], mode='markers', marker=dict(color="black", symbol="x", size=10),name='Change Points')) # We remove the last value from results because it is the last value of the data
        ylabel =  f'{region_name} ' + self.dict_item[y_name]
        xlabel = self.dict_item[x_name]
        fig.update_layout(
        title=f'{ylabel} with Pelt Change Points',
        xaxis_title=xlabel,
        yaxis_title=ylabel)
        return fig


    ## Plot Common Pelt Points for regions
    def gen_common_graph(self, df,  x_name,y_name, commons):
        regions = df['region'].unique()
        color_map = {k:v for k,v in zip(regions, ['red','blue','green'])}
        fig = go.Figure()
        y_values = list(df[y_name])
        x_values = list(df[x_name])
        for region, color in color_map.items():
            x_region = df[df['region'] == region][x_name].values
            y_region = df[df['region'] == region][y_name].values
            fig.add_trace(go.Scatter(x=x_region, y=y_region, mode='lines', name=f'{region}', line=dict(color=color)))
        fig.add_trace(go.Scatter(x=commons, y=[y_values[x_values.index(cp)] for cp in commons], mode='markers', marker=dict(color="black", symbol="x", size=10),name='Change Points'))
        ylabel = self.dict_item[y_name]
        xlabel = self.dict_item[x_name]
        fig.update_layout(
        title=f'{ylabel} with Common Pelt Change Points',
        xaxis_title=xlabel,
        yaxis_title=ylabel)
        return fig 


    ## Get Pelt change Points for region and take the unique values only
    def plot_common_change_points(self, dfs,main_df,x_name, y_name):
        sets = []
        for df in dfs:
            changes =  self.get_change_points(df, x_name, y_name)
            sets.append(set(changes))
        common_values = [sets[x] & sets[x+1] for x in range(len(sets) - 1)]
        commons = [list(x) for x in common_values][0]
        return self.gen_common_graph(main_df, x_name, y_name, commons)


    #3. RDD
    
    #Create RDD Model & return model if needed, otherwise it will return the dataframe, the before and after
    def gen_rdd_model_data(self,df,x_name, y_name, model=False):
        model_ = f'{y_name}~ post_event + {x_name}'
        #event_date_numeric = df[df['date'] == self.event_date]['time_numeric'].values[0]
        df['post_event'] = (df[x_name] >= self.cutoff_date).astype(int)
        if model:
            rdd_model = ols(model_, data=df).fit()
            return rdd_model
        before_event,after_event = self.get_before_after(df)
        #fig = go.Figure()
        return df, before_event,after_event

    #Create Polynomial Lines for before / after events
    def gen_lines_before_after(self,df,x_name,y_name,z_val,degree,id,fig=None,linear=False):
        if not fig:
            fig = go.Figure()
        df,before,after = self.gen_rdd_model_data(df,x_name,y_name,model=False)
        fig = self.gen_poly( fig, before, x_name, y_name,degree,id)
        fig.data[-1].name = f'Before - {z_val}'
        fig = self.gen_poly( fig, after, x_name, y_name,degree,id)
        fig.data[-1].name = f'After - {z_val}'
        if linear:
            fig = self.gen_linear( fig, before, x_name, y_name,id)
            fig.data[-1].name = f'Before Linear - {z_val}'
            fig = self.gen_linear( fig, after, x_name, y_name,id)
            fig.data[-1].name = f'After Linear - {z_val}'
        return fig


    ## Model and Plot RDD
    def plotly_model_rdd(self, df, x_name, y_name, z_val,degree, id,fig=None,update=0,linear=False,size=6,add_line=True):
        """Plots RDD Model
        Args:
            df - Dataframe 
            x_name - Running Variable
            y_name - Treatment variable
            z_val - The RDD being generated for (Region)
            degree - Polynomial degree to use
            id - ID for Colors
            update - Update x axis 
            linear - Add Lines of Best fit 
            size - Size of scatter plot
            add_line - Add the event line"""
        if not fig:
            fig = go.Figure()
        fig = self.gen_scatter( fig, df, x_name, y_name,id, size=size)
        fig.data[-1].name = self.dict_item[y_name] + f' {z_val}'
        fig = self.gen_lines_before_after(df,x_name,y_name,z_val,degree,id,fig=fig,linear=linear)
        if add_line:
            fig = self.add_event_line( fig,df, x_name)
        if update:
            fig.update_xaxes(dtick=update)
        fig.update_layout(
        title=f'RDD Analysis for {z_val} Search Volume for {self.keyword}',
        xaxis_title='Days elapsed since event',
        yaxis_title=self.dict_item[y_name])
        return fig
    
    ### Model and Plot RDD but for multiple dfs into one figure
    def multi_plot_rdd(self, dfs,x_name, y_name, z_name,degree, id,update=0,linear=False,size=6,add_line=True):
        z_vals = []
        fig = go.Figure()
        for i,df in enumerate(dfs):
            fig = self.gen_scatter( fig, df, x_name, y_name,i, size=size)
        for i,df in enumerate(dfs):
            z_val = df[z_name].values[0]
            z_vals.append(z_val)
            i*=2
            fig = self.gen_lines_before_after(df,x_name,y_name,z_val,degree,i,fig=fig,linear=linear)
        if add_line:
            fig = self.add_event_line( fig,df, x_name)
        if update:
            fig.update_xaxes(dtick=update)
        fig.update_layout(
            title=f'Comparing RDD Results for {", ".join(z_vals)} -  Search Volume for "{self.keyword}"',
            xaxis_title='Days elapsed since event',
            yaxis_title=self.dict_item[y_name]
        )
        return fig



    # 4. DID
    def make_did_model(self,df, x_name, y_name, treatment_col, treatment,x_val=None):
        """Creates DID Model
        Args:
            df: Dataframe
            x_name : running variable
            y_name : value you are running reggression on
            treatment_col: Treatment Column (ex - region/ year/ brand)
            treatment: The treatment group
            x_val (optional) : The cutoff point """
        if not x_val:
            x_val = df[df['date'] == self.event_date][x_name].values[0]
        df['post_event'] = np.where(df[x_name]>=x_val,1,0)
        df['treatment'] = np.where(df[treatment_col]==treatment,1,0)
        df['post_treatment']=df['post_event']*df['treatment']
        did_model = ols(f'{y_name} ~  post_event + treatment + post_treatment', df).fit()
        df['fittedvalues'] = did_model.predict()
        return df, did_model


    def plot_did_data_with_control(self,df,x_name,x_val,y_name,treatment_col,degree,color_map=None,title=None):
        """Plots DID with Control
        Args:
            df : DataFrame 
            x_name: The running variable 
            x_val: The value of the Cutoff Point
            y_name: The variable we are doing Did on
            treatment_col: The treatment column (ex - region / year / brand)
            degree: The degree to use for the polynomial 
            color_map : ColorMap to use
            title: the title (Optional)"""
        if not color_map:
            color_map = ['lightsteelblue','darkblue','blue','peachpuff','peru','orangered']
        fig = go.Figure()
        control_group = df[df['treatment'] == 0]
        treatment_group = df[df['treatment'] == 1]
        control_group_before = control_group[control_group['post_event'] == 0 ]
        treatment_group_before = treatment_group[treatment_group['post_event'] == 0 ]
        control_group_after = control_group[control_group['post_event'] == 1 ]
        treatment_group_after= treatment_group[treatment_group['post_event'] == 1 ]
        control = control_group[treatment_col].unique()[0]
        treatment = treatment_group[treatment_col].unique()[0]
        control_group_before = self.poly_eval(control_group_before,x_name,y_name,degree)
        treatment_group_before = self.poly_eval(treatment_group_before,x_name,y_name,degree)
        control_group_after = self.poly_eval(control_group_after,x_name,y_name,degree)
        treatment_group_after = self.poly_eval(treatment_group_after,x_name,y_name,degree)
        fig.add_trace(px.scatter(control_group, x=x_name, y=y_name, color_discrete_sequence=[color_map[0]]).data[0])
        fig.add_trace(go.Scatter(x=control_group_before[x_name], y=control_group_before['dg'], line=dict(color=color_map[1]), mode='lines', name=f'Before - {control}'))
        fig.add_trace(go.Scatter(x=control_group_after[x_name], y=control_group_after['dg'], line=dict(color=color_map[2]), mode='lines', name=f'After - {control}'))
        fig.add_trace(px.scatter(treatment_group, x=x_name, y=y_name, color_discrete_sequence=[color_map[3]]).data[0])
        fig.add_trace(go.Scatter(x=treatment_group_before[x_name], y=treatment_group_before['dg'], line=dict(color=color_map[4]), mode='lines', name=f'Before - {treatment}'))
        fig.add_trace(go.Scatter(x=treatment_group_after[x_name], y=treatment_group_after['dg'], line=dict(color=color_map[5]), mode='lines', name=f'After - {treatment}'))
        fig.add_shape(go.layout.Shape(type="line", x0=x_val, x1=x_val, y0=0, y1=1, xref="x", yref="paper", line=dict(color="black", dash="dash")))
        fig.update_layout(
                    title=f'DID Results for {control} & {treatment} -  Search Volume for "{self.keyword}"' if not title else title,
                    xaxis_title='Days elapsed since event',
                    yaxis_title=self.dict_item[y_name]
                )
        return fig 


    def plot_did_data_treatment(self,df,x_name,x_val,y_name,treatment_col,degree,color_map=None,title=None):
        """Plots DID for Treatment Only
        Args:
            df : DataFrame 
            x_name: The running variable 
            x_val: The value of the Cutoff Point
            y_name: The variable we are doing Did on
            treatment_col: The treatment column (ex - region / year / brand)
            degree: The degree to use for the polynomial 
            color_map : ColorMap to use
            title: the title (Optional)"""
        if not color_map:
            color_map = ['lightsteelblue','darkblue','blue','peachpuff','peru','orangered']
        fig = go.Figure()
        treatment_group = df[df['treatment'] == 1]
        treatment_group_before = treatment_group[treatment_group['post_event'] == 0 ]
        treatment_group_after= treatment_group[treatment_group['post_event'] == 1 ]
        treatment = treatment_group[treatment_col].unique()[0]
        treatment_group_before = self.poly_eval(treatment_group_before,x_name,y_name,degree)
        treatment_group_after = self.poly_eval(treatment_group_after,x_name,y_name,degree)
        fig.add_trace(px.scatter(treatment_group, x=x_name, y=y_name, color_discrete_sequence=[color_map[3]]).data[0])
        fig.add_trace(go.Scatter(x=treatment_group_before[x_name], y=treatment_group_before['dg'], line=dict(color=color_map[4]), mode='lines', name=f'Before - {treatment}'))
        fig.add_trace(go.Scatter(x=treatment_group_after[x_name], y=treatment_group_after['dg'], line=dict(color=color_map[5]), mode='lines', name=f'After - {treatment}'))
        fig.add_shape(go.layout.Shape(type="line", x0=x_val, x1=x_val, y0=0, y1=1, xref="x", yref="paper", line=dict(color="black", dash="dash")))
        fig.update_layout(
                    title=f'DID Results for {treatment} -  Search Volume for "{self.keyword}"' if not title else title,
                    xaxis_title=self.dict_item[x_name],
                    yaxis_title=self.dict_item[y_name]
                )
        return fig 


    # 5. Maps
    #Make new column with the geo values
    def countries_to_iso(self,df):
        df['id'] = coco.convert(names=df['Country'],to='ISO3')
        return df.sort_values(by='date',ascending=True)
    
    def rename_states(self,df):
        #https://gist.github.com/rogerallen/1583593
        us_state_to_abbrev = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC', 'American Samoa': 'AS', 'Guam': 'GU', 'Northern Mariana Islands': 'MP', 'Puerto Rico': 'PR', 'United States Minor Outlying Islands': 'UM', 'U.S. Virgin Islands': 'VI'}
        df['state'] = df['Region'].apply(lambda x: us_state_to_abbrev[x])
        return df.sort_values(by='date',ascending=True)
    
    # Make World Map 
    def make_world_map(self, df):
        df = self.countries_to_iso(df)
        fig = px.choropleth(df, 
                    locations='id',
                    color='raw_daily',
                    hover_name='Country',
                    animation_frame = df['date'],
                    color_continuous_scale=px.colors.sequential.thermal[::-1],
                    height = int(800),
                    range_color=[0, 100],
                    title = f'Worldwide Searches for "{self.keyword}" +/- 15 days from event')
        return fig
    
    #Make US Map 
    def make_us_map(self,df):
        df = self.rename_states(df)
        fig = px.choropleth(df,
                    locations='state',
                    locationmode='USA-states',
                    scope='usa',
                    color='raw_daily',
                    hover_name='state',
                    animation_frame = df['date'],
                    color_continuous_scale=px.colors.sequential.thermal[::-1],
                    height = int(800),
                    range_color=[0, 100],
                    title = f'US Searches for "{self.keyword}" +/- 15 days from event')
        return fig
    
    #6. Additional Methods
    ## Harmonize like Stata code
    def quick_harmo(self, df): #pandas version of the stata code
        df['week_of'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='d')
        df['calc_weekly'] = df.groupby('week_of')['raw_daily'].transform('mean')
        df['weight'] = df['raw_weekly'] / df['calc_weekly']
        df['weighted_daily'] = df['raw_daily'] * df['weight']
        max_weighted_daily = df['weighted_daily'].max()
        df['max_weighted_daily'] = max_weighted_daily
        df['harmonized_daily_rescaled'] = (df['weighted_daily'] / max_weighted_daily) * 100
        df.drop(columns=['week_of', 'calc_weekly', 'weight', 'weighted_daily', 'max_weighted_daily'], inplace=True)
        return df