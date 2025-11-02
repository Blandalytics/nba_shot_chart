import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time

from matplotlib.gridspec import GridSpec
from nba_api.stats.endpoints import PlayerGameLogs
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players
from scipy.stats import gaussian_kde

## Set Styling
# Plot Style
pl_white = '#FEFEFE'
pl_background = '#162B50'
pl_text = '#72a3f7'
pl_line_color = '#293a6b'

fpath = os.path.join(os.getcwd(), 'Arimo/Arimo-Regular.ttf')
prop = fm.FontProperties(fname=fpath)

sns.set(font='Arimo')
sns.set_theme(
    style={
        'axes.edgecolor': pl_background,
        'axes.facecolor': pl_background,
        'axes.labelcolor': pl_white,
        'xtick.color': pl_white,
        'ytick.color': pl_white,
        'figure.facecolor':pl_background,
        'grid.color': pl_background,
        'grid.linestyle': '-',
        'legend.facecolor':pl_background,
        'text.color': pl_white
     },
    )

nba_players = players.get_active_players()

st.set_page_config(page_title='NBA Shot Chart', page_icon='🏀',layout='wide')
st.title('NBA Shot Chart')
st.write("NBA players earn points through a combination of taking shots (and free throws), taking quality shots, and making those shots (and free throws). This app is designed to illustrate *how* a player scored (or didn't score) points in a given game, and visually represent the quality of their shot selection.")
st.write("The expected FG% model was trained on 2022-23, 2023-24, and 2024-25 data. If you're interested in the xFG% value of each X,Y, coordinate, a csv can be found [here](https://github.com/Blandalytics/nba_shot_chart/blob/main/nba_xFG_values.csv).")
st.write('Find me [@Blandalytics](https://bsky.app/profile/blandalytics.pitcherlist.com), and subscribe to [Pitcher List](https://pitcherlist.com/premium/) if you want to support my (mostly baseball) work!')

@st.cache_data(ttl=1200,show_spinner=f"Loading shots")
def load_season(year='2025-26'):
    season_df = shotchartdetail.ShotChartDetail(
        team_id = 0, # can input the id# but 0, will return all
        player_id = 0, # can input the id# but 0, will return all
        context_measure_simple = 'FGA', # also 'PTS' has ONLY makes
        season_nullable = year,
        season_type_all_star = 'Regular Season').get_data_frames()[0]
    season_df['GAME_DATE'] = pd.to_datetime(season_df['GAME_DATE']).dt.date
    season_df['TIME_REMAINING'] = season_df['MINUTES_REMAINING'].add(season_df['SECONDS_REMAINING'].div(60))
    season_df['last_5_sec'] = np.where(season_df['TIME_REMAINING']<=1/12,1,0)
    season_df['last_5_sec'] = season_df['last_5_sec'].astype('category').cat.codes
    
    season_df['SHOT_MADE_FLAG'] = season_df['SHOT_MADE_FLAG'].astype('category').cat.codes
    
    season_df['SHOT_PTS'] = season_df['SHOT_TYPE'].map({
        '3PT Field Goal':3,
        '2PT Field Goal':2
    }).mul(season_df['SHOT_MADE_FLAG'])
    
    blank_df = pd.read_csv('nba_xFG_values.csv')
    
    season_df['SHOT_DISTANCE_calc'] = (season_df['LOC_X'].abs()**2 + season_df['LOC_Y'].abs()**2) ** 0.5 / 10
    blank_df['SHOT_DISTANCE_calc'] = (blank_df['LOC_X'].abs()**2 + blank_df['LOC_Y'].abs()**2) ** 0.5 / 10
    
    season_df[['xFG%','xPTS']] = season_df[['LOC_X','LOC_Y']].merge(blank_df,
                                                                    how='left',
                                                                    on=['LOC_X','LOC_Y'])[['xFG%','xPTS']]
    season_df['LOC_X'] = season_df['LOC_X'].mul(-1) # Axis is flipped
    
    season_df['xFG%_avg'] = season_df['SHOT_MADE_FLAG'].groupby([season_df['SHOT_DISTANCE'],season_df['SHOT_TYPE'],season_df['last_5_sec']]).transform('mean')
    season_df['xPTS_avg'] = season_df['xFG%_avg'].mul(season_df['SHOT_TYPE'].map({
        '3PT Field Goal':3,
        '2PT Field Goal':2
    }))

    ft_df = PlayerGameLogs(
        season_nullable = year, # change year(s) if needed
        season_type_nullable = 'Regular Season' # Regular Season, Playoffs, Pre Season
        )
    ft_df = ft_df.get_data_frames()[0]
    season_df = season_df.merge(ft_df[['PLAYER_ID','GAME_ID','FTM', 'FTA']],
                                how='left',on=['PLAYER_ID','GAME_ID'])
    season_df[['FTM', 'FTA']] = season_df[['FTM', 'FTA']].fillna(0)
    season_df['FTM'] = season_df['FTM'].div(season_df['SHOT_ATTEMPTED_FLAG'].groupby([season_df['PLAYER_ID'],season_df['GAME_ID']]).transform('count'))
    season_df['FTA'] = season_df['FTA'].div(season_df['SHOT_ATTEMPTED_FLAG'].groupby([season_df['PLAYER_ID'],season_df['GAME_ID']]).transform('count'))

    season_df['GAME_PLAYED'] = 1 / season_df['SHOT_ATTEMPTED_FLAG'].groupby([season_df['PLAYER_ID'],season_df['GAME_ID']]).transform('count')
    season_df['PTS'] = season_df[['SHOT_PTS','FTM']].sum(axis=1)
    
    center_hoop = 12.5
    background_data = (
        blank_df
        .loc[(blank_df['SHOT_DISTANCE_calc']<=35) &
                ((blank_df['LOC_Y']>=-center_hoop/2) | 
                 (blank_df['LOC_X'].abs()>=80) | 
                 (blank_df['LOC_X'].abs().sub(30)>blank_df['LOC_Y'].abs().sub(center_hoop).mul(1.5)))]
        .copy()
    )
    return season_df.loc[season_df['PLAYER_ID'].isin([x['id'] for x in nba_players])], background_data

# year = st.selectbox('Select a Season:',['2025-26','2024-25','2023-24','2022-23'], index=0)

season_df, background_data = load_season()
pts_per_shot = 1.09
pts_per_ft = 190241 / 245985 # 2022-23 to 2024-25 avg

pad1, col1, col2, pad2 = st.columns([0.2,0.3,0.3,0.2])
with col1:
    player_name = st.selectbox('Select a player',list(season_df.groupby('PLAYER_NAME')['PTS'].sum().sort_values(ascending=False).index), index=0)
    player_id = [x['id'] for x in nba_players if x['full_name']==player_name][0]
with col2:
    col3, col4 = st.columns([1/3,2/3])
    with col3:
        season_long = st.toggle('Season-long chart?')
    if season_long:
        game_date = ''
    else:
        with col4:
            game_date = st.selectbox('Select a game',list(season_df.loc[season_df['PLAYER_ID']==player_id,'GAME_DATE'].sort_values(ascending=False).unique()), 
                                     index=0, format_func=lambda x: x.strftime('%-m/%-d/%y'))

def shot_summary(player_id,game_date=game_date, season_long=season_long):
    line_outline='w'
    line_color = 'k'
    
    backboard_depth = 40
    hoop_depth = 52.5
    center_hoop = 12.5
    
    y_lim = 470
    y_adj = backboard_depth+center_hoop

    hue_norm = colors.CenteredNorm(pts_per_shot,0.42)

    if season_long:
        game_data = season_df.loc[(season_df['PLAYER_ID']==player_id)]
    else:
        game_data = season_df.loc[(season_df['PLAYER_ID']==player_id) & (season_df['GAME_DATE']==game_date)]
    # chart_data = shot_chart_detail.loc[(shot_chart_detail['LOC_Y']<=y_lim) & (shot_chart_detail['last_5_sec']==0)].copy()
    fig = plt.figure(figsize=(13,8))
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[5.1,7.9],wspace=0,
                  height_ratios=[7.7,0.3],hspace=0)
    
    ax1 = fig.add_subplot(gs[:, 1])
    hex_mult = 7
    hb = ax1.hexbin(x=background_data['LOC_X'],
              y=background_data['LOC_Y'],
              C=background_data['xPTS'],
              cmap='vlag',
              norm=hue_norm,
              gridsize=(5*hex_mult,3*hex_mult),
              # mincnt=1,
              extent=(-250,250,-y_adj,y_lim))
    
    sns.scatterplot(game_data.loc[(game_data['SHOT_MADE_FLAG']==0)],
                    x='LOC_X',
                    y='LOC_Y',
                    color='None',
                    marker='X',
                    linewidth=3.5,
                    edgecolor='w',
                    s=140,
                    alpha=0.75,
                    legend=False,
                    zorder=11,
                    ax=ax1
                   )
    sns.scatterplot(game_data.loc[(game_data['SHOT_MADE_FLAG']==0)],
                    x='LOC_X',
                    y='LOC_Y',
                    color='None',
                    marker='X',
                    linewidth=1.5,
                    edgecolor='purple',
                    s=125,
                    legend=False,
                    zorder=12,
                    ax=ax1
                   )
    
    sns.scatterplot(game_data.loc[(game_data['SHOT_MADE_FLAG']==1)],
                    x='LOC_X',
                    y='LOC_Y',
                    color='green',
                    marker='o',
                    linewidth=1,
                    edgecolor='w',
                    s=125,
                    legend=False,
                    zorder=12,
                    ax=ax1
                   )
    
    
    hoop1 = mpl.patches.Arc((0,0),15,15,color=line_outline,linewidth=3)
    ax1.add_patch(hoop1)
    hoop2 = mpl.patches.Arc((0,0),15,15,color=line_color,linewidth=1.5)
    ax1.add_patch(hoop2)
    
    hoop_arc1 = mpl.patches.Arc((0,0),80,80,theta1=0,theta2=180,color=line_outline,linewidth=3)
    ax1.add_patch(hoop_arc1)
    hoop_arc2 = mpl.patches.Arc((0,0),80,80,theta1=0,theta2=180,color=line_color,linewidth=1.5)
    ax1.add_patch(hoop_arc2)
    
    # 3pt Line
    three_point_arc1 = mpl.patches.Arc((0,0),475,475,theta1=22,theta2=158,color=line_outline,linewidth=3,zorder=9)
    ax1.add_patch(three_point_arc1)
    ax1.plot([-220,-220],[0-y_adj,141-y_adj],color=line_outline,linewidth=3,zorder=9)
    ax1.plot([220,220],[0-y_adj,141-y_adj],color=line_outline,linewidth=3,zorder=9)
    
    three_point_arc2 = mpl.patches.Arc((0,0),475,475,theta1=22,theta2=158,color=line_color,linewidth=1.5,zorder=10)
    ax1.add_patch(three_point_arc2)
    ax1.plot([-220,-220],[0-y_adj,141-y_adj],color=line_color,linewidth=1.5,zorder=10)
    ax1.plot([220,220],[0-y_adj,141-y_adj],color=line_color,linewidth=1.5,zorder=10)
    
    # Free Throw Circle
    free_throw_full = mpl.patches.Arc((0,190-y_adj),120,120,theta1=0,theta2=360,color=line_outline,linewidth=3,zorder=9)
    ax1.add_patch(free_throw_full)
    free_throw_top = mpl.patches.Arc((0,190-y_adj),120,120,theta1=0,theta2=180,color=line_color,linewidth=1.5,zorder=10)
    ax1.add_patch(free_throw_top)
    free_throw_bot = mpl.patches.Arc((0,190-y_adj),120,120,theta1=180,theta2=360,linestyle='--',color=line_color,linewidth=1.5,zorder=10)#(0, (5, 10)))
    ax1.add_patch(free_throw_bot)
    
    #Backboard
    ax1.plot([-30,30],[40-y_adj,40-y_adj],color=line_color,linewidth=3)
    ax1.plot([-30,30],[40-y_adj,40-y_adj],color=line_outline,linewidth=1.5)
    
    #Free Throw box
    ax1.plot([-80,80],[190-y_adj,190-y_adj],color=line_outline,linewidth=3,zorder=9)
    ax1.plot([-80,-80],[0-y_adj,190-y_adj],color=line_outline,linewidth=3,zorder=9)
    ax1.plot([80,80],[0-y_adj,190-y_adj],color=line_outline,linewidth=3,zorder=9)
    
    ax1.plot([-80,80],[190-y_adj,190-y_adj],color=line_color,linewidth=1.5,zorder=10)
    ax1.plot([-80,-80],[0-y_adj,190-y_adj],color=line_color,linewidth=1.5,zorder=10)
    ax1.plot([80,80],[0-y_adj,190-y_adj],color=line_color,linewidth=1.5,zorder=10)
    
    #Baseline
    ax1.plot([-250,250],[0-y_adj,0-y_adj],color=line_outline,linewidth=1.5,zorder=9)
    ax1.plot([-249,-249],[0-y_adj,y_lim],color=line_outline,linewidth=1.5,zorder=9)
    ax1.plot([249,249],[0-y_adj,y_lim],color=line_outline,linewidth=1.5,zorder=9)
    
    ax1.set(xlim=(-250,250),ylim=(-53,400),aspect=1)
    ax1.set_axis_off()
    ax1.text(242.5,385,'@blandalytics',ha='right',fontweight='light')
    
    cb = fig.colorbar(hb, location='top',orientation='horizontal',
                      ax=ax1, label='',shrink=0.94, panchor=(0.5,0),
                     pad=0)
    cb.outline.set_color('white')
    cb.outline.set_linewidth(1.5)
    cb.ax.set(xlim=(0.69,1.51))
    
    expected_fg_percent = game_data['xFG%'].mean()
    expected_points = game_data['xPTS'].mean()
    actual_points = game_data['SHOT_PTS'].mean()
    
    def normalize(arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)   
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr
    if game_data.shape[0]>1:
        density = gaussian_kde(np.clip(game_data['xPTS'],cb.ax.get_xlim()[0],cb.ax.get_xlim()[1]))
        xs = np.linspace(cb.ax.get_xlim()[0],cb.ax.get_xlim()[1],200)
        fit_xs = xs / max(xs)
        density.covariance_factor = lambda : .1
        density._compute_covariance()
        # cb.ax.plot(xs,density(xs) / max(density(xs)*1.2)+0.05,color='k',alpha=0.75)
        cb.ax.fill_between(xs,density(xs) / max(density(xs)*1.2)+0.05,color='k',alpha=0.4)
        cb.ax.set(xlabel='')
    cb.ax.axvline(expected_points,color='k',linewidth=2)
    cb.ax.axvline(actual_points,
                  color='w',
                  linewidth=3)
    cb.ax.axvline(actual_points,
                  color='g' if actual_points >= expected_points else 'purple',
                  linewidth=1.5)
    if abs(actual_points-expected_points) >= 0.025:
        cb.ax.arrow(expected_points,0.5,
                    actual_points-expected_points,0,
                    width=0.01,
                    length_includes_head=True,
                    color='g' if actual_points >= expected_points else 'purple',
                    edgecolor='w',linewidth=2)
    # cb.ax.axvline(pts_per_shot,color='k',linewidth=0.5,linestyle='--')
    ax1.text(0,475,f'xPoints-per-Shot: {expected_points:.2f}',ha='center',va='center',fontsize=18,fontproperties=prop)
    
    ax2 = fig.add_subplot(gs[:,0])
    
    volume_points = game_data['SHOT_ATTEMPTED_FLAG'].sum() * pts_per_shot
    quality_points = game_data['xPTS'].sub(pts_per_shot).sum()
    finishing_points = game_data['SHOT_PTS'].sub(game_data['xPTS']).sum()
    
    fta_points = game_data['FTA'].sum() * pts_per_ft
    ftm_points = game_data['FTM'].sum() - fta_points
    total_points = int(round(volume_points + quality_points + finishing_points + fta_points + ftm_points,0))
    
    categories = ['Shot\nVolume','Shot\nQuality','Shot\nMaking','FT\nAttempts','FT\nMakes']
    values = [volume_points,quality_points,finishing_points,fta_points,ftm_points]
    cumulative_values = np.cumsum(values)
    max_val = max(cumulative_values)
    
    for i in range(len(values)):
        if values[i] >= 0:
            color = 'green'
        else:
            color = 'purple'
        ax2.bar(
            categories[i],
            values[i],
            bottom=cumulative_values[i] - values[i],
            color=color,
            edgecolor=color,
            width=0.8
        )
        ax2.text(categories[i],max_val*1.0675,f'{categories[i]:}',
                 fontsize=11,
                ha='center',va='center',color='w')
        ax2.text(categories[i],cumulative_values[i],f'{values[i]:+.1f}',
                 fontsize=10 if season_long else 12,
                ha='center',va='center',color=color,fontweight='bold',
                bbox=dict(boxstyle='round', fc='w', ec=color))
        
    xlim = (-1/(len(categories)-1),len(categories)-1+2/(len(categories)))
    ax2.set(xlim=xlim)
    x_width = xlim[1] - xlim[0]
    for i in range(len(cumulative_values)):
        ax2.plot([i+0.02+(x_width/3+0.2) / x_width,
                 i-0.02 + (2*x_width/3-0.2) / x_width],
                [cumulative_values[i],cumulative_values[i]],
                 linestyle=(0, (1, 1)),
                color='w')
    points_scored = int(round(sum(values),0))
    # ax.text(xlim[1]*1.025,cumulative_values[-1],f'{points_scored}pts',color='w',ha='left',va='center')
    ax2.tick_params(axis='both', which='both',length=0)
    ax2.axhline(points_scored,color='w',
                xmin=xlim[0]+0.31,
                xmax=(xlim[1]+0.4) / x_width)
    ax2.text(2,max_val*1.13,f'{points_scored} Points Scored',ha='center',fontsize=18)
    ax2.axis('off')
    ax2.set(xlim=(xlim[0]-0.4,xlim[1]))
    ax2.set_ylim([0,max_val*1.15])
    
    player_name = game_data['PLAYER_NAME'].iloc[0]
    if season_long:
        fig.suptitle(f'Shot Summary: {player_name} (2025-26)\n',fontsize=24, 
                     x=0.51,y=0.91,ha='center',va='center',fontproperties=prop)
    else:
        date_text = game_date.strftime('%#m/%#d/%y')
        fig.suptitle(f'Shot Summary: {player_name} ({date_text})\n',fontsize=24, 
                     x=0.51,y=0.91,ha='center',va='center',fontproperties=prop)
    
    sns.despine()
    st.pyplot(fig,width='content')

pad1, col1, pad2 = st.columns([0.225,0.55,0.225])
with col1:
    shot_summary(player_id)

col1, col2, col3 = st.columns(3)
with col1:
    per_shot = st.toggle('Calculate stats per-shot?')
    
with col2:
    if per_shot:
        season_thresh = st.slider('Min shot attempts in season',
                  min_value=1,
                  max_value=season_df.groupby('PLAYER_ID')['SHOT_ATTEMPTED_FLAG'].sum().max(),
                  value=50
                 )
with col3:
    if per_shot:
        game_thresh = st.slider('Min shot attempts in game',
                  min_value=1,
                  max_value=season_df.groupby(['PLAYER_ID','GAME_ID'])['SHOT_ATTEMPTED_FLAG'].sum().max(),
                  value=5
                 )
        
if per_shot:
    attempt_df = (
        season_df
        .loc[season_df['SHOT_ATTEMPTED_FLAG'].groupby(season_df['PLAYER_ID']).transform('sum') >= season_thresh]
        .assign(volume_points = lambda x: x['SHOT_ATTEMPTED_FLAG'].mul(pts_per_shot),
                quality_points = lambda x: x['xPTS'].sub(x['SHOT_ATTEMPTED_FLAG'].mul(pts_per_shot)),
                making_points = lambda x: x['SHOT_PTS'].sub(x['xPTS']))
        .rename(columns={
            'PLAYER_NAME':'Player',
            'SHOT_ATTEMPTED_FLAG':'Shots',
            'SHOT_PTS':'Points',
            'volume_points':'Volume Pts',
            'quality_points':'Quality Pts',
            'making_points':'Make Pts'
        })
        .groupby('Player')
        [['Points','Volume Pts','Quality Pts','Make Pts']]
        .mean()
        .round(2)
        .sort_values('Points',ascending=False)
    )
    game_df = (
        season_df
        .loc[season_df['SHOT_ATTEMPTED_FLAG'].groupby([season_df['PLAYER_ID'],season_df['GAME_DATE']]).transform('sum') >= game_thresh]
        .assign(volume_points = lambda x: x['SHOT_ATTEMPTED_FLAG'].mul(pts_per_shot),
                quality_points = lambda x: x['xPTS'].sub(x['SHOT_ATTEMPTED_FLAG'].mul(pts_per_shot)),
                making_points = lambda x: x['SHOT_PTS'].sub(x['xPTS']))
        .rename(columns={
            'PLAYER_NAME':'Player',
            'GAME_DATE':'Date',
            'SHOT_ATTEMPTED_FLAG':'Shots',
            'SHOT_PTS':'Points',
            'volume_points':'Volume Pts',
            'quality_points':'Quality Pts',
            'making_points':'Make Pts'
        })
        .groupby(['Player','Date'])
        [['Points','Volume Pts','Quality Pts','Make Pts']]
        .mean()
        .round(2)
        .sort_values('Points',ascending=False)
    )
else:
    attempt_df = (
        season_df
        .assign(volume_points = lambda x: x['SHOT_ATTEMPTED_FLAG'].mul(pts_per_shot),
                quality_points = lambda x: x['xPTS'].sub(x['SHOT_ATTEMPTED_FLAG'].mul(pts_per_shot)),
                making_points = lambda x: x['SHOT_PTS'].sub(x['xPTS']),
                fta_points = lambda x: x['FTA'].mul(pts_per_ft),
                ftm_points = lambda x: x['FTM'].sub(x['FTA'].mul(pts_per_ft)))
        .rename(columns={
            'PLAYER_NAME':'Player',
            'GAME_PLAYED':'G',
            'SHOT_ATTEMPTED_FLAG':'FGA',
            'volume_points':'Vol Pts',
            'quality_points':'Qual Pts',
            'making_points':'Make Pts',
            'SHOT_PTS':'FG Pts',
            'fta_points':'FTA Pts',
            'ftm_points':'FT Make Pts',
            'FTM':'FT Pts',
            'PTS':'Pts',
        })
        .groupby('Player')
        [['G','Pts','FGA','Vol Pts','Qual Pts','Make Pts','FG Pts','FTA Pts','FT Make Pts','FT Pts']]
        .sum()
        .astype({
            'G':'int','FGA':'int','Pts':'int','FG Pts':'int','FT Pts':'int'
        })
        .round(1)
        .sort_values('Pts',ascending=False)
    )
    
    game_df = (
        season_df
        .assign(volume_points = lambda x: x['SHOT_ATTEMPTED_FLAG'].mul(pts_per_shot),
                quality_points = lambda x: x['xPTS'].sub(x['SHOT_ATTEMPTED_FLAG'].mul(pts_per_shot)),
                making_points = lambda x: x['SHOT_PTS'].sub(x['xPTS']),
                fta_points = lambda x: x['FTA'].mul(pts_per_ft),
                ftm_points = lambda x: x['FTM'].sub(x['FTA'].mul(pts_per_ft)))
        .rename(columns={
            'PLAYER_NAME':'Player',
            'SHOT_ATTEMPTED_FLAG':'Shots',
            'GAME_DATE':'Date',
            'SHOT_ATTEMPTED_FLAG':'FGA',
            'volume_points':'Vol Pts',
            'quality_points':'Qual Pts',
            'making_points':'Make Pts',
            'SHOT_PTS':'FG Pts',
            'fta_points':'FTA Pts',
            'ftm_points':'FT Make Pts',
            'FTM':'FT Pts',
            'PTS':'Pts',
        })
        .groupby(['Player','Date'])
        [['Pts','FGA','Vol Pts','Qual Pts','Make Pts','FG Pts','FTA Pts','FT Make Pts','FT Pts']]
        .sum()
        .astype({
            'FGA':'int','Pts':'int','FG Pts':'int','FT Pts':'int'
        })
        .round(2)
        .sort_values('Pts',ascending=False)
    )

col1, col2 = st.columns(2)
with col1:
    st.header('Season Leaderboard')
    st.dataframe(attempt_df)
with col2:
    st.header('Game Leaderboard')
    st.dataframe(game_df)

team_map = {
 'Atlanta Hawks': 1610612737,
 'Boston Celtics': 1610612738,
 'Brooklyn Nets': 1610612751,
 'Charlotte Hornets': 1610612766,
 'Chicago Bulls': 1610612741,
 'Cleveland Cavaliers': 1610612739,
 'Dallas Mavericks': 1610612742,
 'Denver Nuggets': 1610612743,
 'Detroit Pistons': 1610612765,
 'Golden State Warriors': 1610612744,
 'Houston Rockets': 1610612745,
 'Indiana Pacers': 1610612754,
 'LA Clippers': 1610612746,
 'Los Angeles Lakers': 1610612747,
 'Memphis Grizzlies': 1610612763,
 'Miami Heat': 1610612748,
 'Milwaukee Bucks': 1610612749,
 'Minnesota Timberwolves': 1610612750,
 'New Orleans Pelicans': 1610612740,
 'New York Knicks': 1610612752,
 'Oklahoma City Thunder': 1610612760,
 'Orlando Magic': 1610612753,
 'Philadelphia 76ers': 1610612755,
 'Phoenix Suns': 1610612756,
 'Portland Trail Blazers': 1610612757,
 'Sacramento Kings': 1610612758,
 'San Antonio Spurs': 1610612759,
 'Toronto Raptors': 1610612761,
 'Utah Jazz': 1610612762,
 'Washington Wizards': 1610612764
}

team_colors = {
    'Boston Celtics':{
        'background':'#007a33',
        'text':'white'
    },
    'Brooklyn Nets':{
        'background':'k',
        'text':'white'
    },
    'New York Knicks':{
        'background':'#f58426',
        'text':'white'
    },
    'Philadelphia 76ers':{
        'background':'#002b5c',
        'text':'white'
    },
    'Toronto Raptors':{
        'background':'#ce1141',
        'text':'white'
    },
    'Chicago Bulls':{
        'background':'#ce1141',
        'text':'white'
    },
    'Cleveland Cavaliers':{
        'background':'#860038',
        'text':'white'
    },
    'Detroit Pistons':{
        'background':'#c8102e',
        'text':'white'
    },
    'Indiana Pacers':{
        'background':'#002d62',
        'text':'white'
    },
    'Milwaukee Bucks':{
        'background':'#00471b',
        'text':'white'
    },
    'Atlanta Hawks':{
        'background':'#e03a3e',
        'text':'white'
    },
    'Charlotte Hornets':{
        'background':'#00788c',
        'text':'white'
    },
    'Miami Heat':{
        'background':'#98002e',
        'text':'white'
    },
    'Orlando Magic':{
        'background':'#0077c0',
        'text':'white'
    },
    'Washington Wizards':{
        'background':'#002b5c',
        'text':'white'
    },
    'Denver Nuggets':{
        'background':'#0e2240',
        'text':'white'
    },
    'Minnesota Timberwolves':{
        'background':'#0c2340',
        'text':'white'
    },
    'Oklahoma City Thunder':{
        'background':'#007ac1',
        'text':'white'
    },
    'Portland Trail Blazers':{
        'background':'#e03a3e',
        'text':'white'
    },
    'Utah Jazz':{
        'background':'#002b5c',
        'text':'white'
    },
    'Golden State Warriors':{
        'background':'#1d428a',
        'text':'white'
    },
    'LA Clippers':{
        'background':'#c8102e',
        'text':'white'
    },
    'Los Angeles Lakers':{
        'background':'#552583',
        'text':'white'
    },
    'Phoenix Suns':{
        'background':'#1d1160',
        'text':'white'
    },
    'Sacramento Kings':{
        'background':'#5a2d81',
        'text':'white'
    },
    'Dallas Mavericks':{
        'background':'#00538c',
        'text':'white'
    },
    'Houston Rockets':{
        'background':'#ce1141',
        'text':'white'
    },
    'Memphis Grizzlies':{
        'background':'#5d76a9',
        'text':'white'
    },
    'New Orleans Pelicans':{
        'background':'#0c2340',
        'text':'white'
    },
    'San Antonio Spurs':{
        'background':'#c4ced4',
        'text':'#000000'
    }
}

@st.cache_data(ttl=1200,show_spinner=f"Loading minutes")
def load_league_minutes(season='2025-26'):
    df_player_game_logs = PlayerGameLogs(
        league_id_nullable ='00', # NBA
        season_nullable = season, # change year(s) if needed
        season_type_nullable = 'Regular Season' # Regular Season, Playoffs, Pre Season
        ).get_data_frames()[0]

    df_player_game_logs['short_date'] = pd.to_datetime(df_player_game_logs['GAME_DATE']).dt.strftime('%#b %#d')

    df_player_game_logs['team_game']  = df_player_game_logs.groupby("TEAM_ID")["GAME_DATE"].rank(method="dense", ascending=False)
    df_player_game_logs['player_game'] = df_player_game_logs.groupby("PLAYER_ID")["GAME_DATE"].rank(method="dense", ascending=False)

    df_player_game_logs['Last 3'] = np.where(df_player_game_logs['team_game']<=3,df_player_game_logs['MIN'],None)
    df_player_game_logs['Last 5'] = np.where(df_player_game_logs['team_game']<=5,df_player_game_logs['MIN'],None)
    df_player_game_logs['Last 10'] = np.where(df_player_game_logs['team_game']<=10,df_player_game_logs['MIN'],None)

    return df_player_game_logs[['TEAM_ID','PLAYER_NAME','team_game','GAME_DATE','short_date','MIN','Last 3','Last 5','Last 10']]

league_minutes = load_league_minutes()

def team_minutes(team_name,df_player_game_logs=league_minutes):
    team_id = team_map[team_name]
    last_5_df = pd.pivot_table(df_player_game_logs.loc[(df_player_game_logs['TEAM_ID']==team_id) & 
                                                       (df_player_game_logs['team_game']<=5)],
                               index=['PLAYER_NAME'],
                               values=['MIN'],
                               columns=['GAME_DATE'])
    last_5_df.columns = df_player_game_logs.loc[(df_player_game_logs['TEAM_ID']==team_id) & 
                                                       (df_player_game_logs['team_game']<=5),
                                                       'short_date'].unique()[::-1]

    team_df = pd.merge(
        df_player_game_logs.loc[df_player_game_logs['TEAM_ID']==team_id].groupby('PLAYER_NAME')[['Last 3','Last 5','Last 10','MIN']].mean().rename(columns={'MIN':'All'}).sort_values(['Last 3','Last 5','Last 10','All'],ascending=False),
        last_5_df[[x for x in last_5_df.columns[::-1]]],
        how='left',
        left_index=True,
        right_index=True).astype('float').round(1)
    team_df.index.name = None
    headers = {
                'selector': 'th',
                'props': f'text-align: center; background-color: {team_colors[TEAM]['background']}; color: {team_colors[TEAM]['text']};'
            }
    return (
        team_df
        .style
        .set_table_styles([headers])
        .format('{:.1f}', na_rep="")
    )

st.title('Team Minutes Breakdown')
all_teams = st.toggle('Show tables for all teams?')

divisions = {
    'Eastern':['Atlantic','Central','Southeast'],
    'Western':['Northwest','Pacific','Southwest'],
}

if all_teams:
    col1, col2 = st.columns(2)
    with col1:
        st.header('Eastern Conference')
        east = col1.container(height=600)
        with east:
            for division in range(3):
                st.header(divisions['Eastern'][division])
                for TEAM in list(team_colors.keys())[division*5:(division+1)*5]:
                    st.subheader(TEAM)
                    team_df = team_minutes(TEAM)
                    st.dataframe(team_df)
    with col2:
        st.header('Western Conference')
        west = col2.container(height=600)
        with west:
            for division in range(3):
                st.header(divisions['Western'][division])
                for TEAM in list(team_colors.keys())[(division+3)*5:(division+4)*5]:
                    st.subheader(TEAM)
                    team_df = team_minutes(TEAM)
                    st.dataframe(team_df)
else:
    TEAM = st.selectbox('Select a team',list(team_map.keys()), index=20)
    team_df = team_minutes(TEAM)
    st.dataframe(team_df)
