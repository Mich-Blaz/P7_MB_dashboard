import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



##### Parametrisation seaborn et matplotlib


plt.rcParams.update({'text.color': "black",
                     'axes.labelcolor': "black"})

plt.rcParams['axes.labelsize'] = 5
plt.rcParams['axes.titlesize'] = 5




# importation des bases de données nécessaires: 
df_col = pd.read_csv("description_col.csv")
df_client = pd.read_csv('d_to_dash_sample.csv').set_index('SK_ID_CURR')
df_shap = pd.read_csv('shapley_importance.csv').set_index('Features').T



##### Config_page
st.set_page_config(
    page_title="Projet7: dashboard Home Credit",
    page_icon="✅",
    layout="wide",
)


###### Sidebar donc seulement ce qui aparait sur le cote


st.sidebar.title('Welcome to Home Credit prediction')

st.sidebar.image: st.sidebar.image("images1.png", use_column_width=True)
st.sidebar.write('Description *Dashboard de visualisation de la prédiction pour octroiement de crédit* :sunglasses: :\n ')


choice_db = st.sidebar.selectbox(
        'Comparaison globale ou locale: ',
        ['Locale','Globale'])
choice_hue = st.sidebar.selectbox(
        'Comparaison TARGET ou NearestNeighbors: ',
        ['TARGET','NearestNeighbors'])
choice_plot = st.sidebar.selectbox(
        'Graph pour les valeurs floattantes: ',
        ['Histogramme','Density'])



st.sidebar.write('Prediction for default risk credit')

sd_b1,sd_b2=st.sidebar.columns(2)

sd_b2.image: sd_b2.image("images.png", use_column_width=True)
sd_b1.image: sd_b1.image("images2.png", use_column_width=True)

st.sidebar.write('Application faite pour le projet 7 OpenclassRooms, formation DataScientist. Lien avec une API disponible sur le github')
st.sidebar.write('Done Michel Blazevic: https://github.com/Mich-Blaz')


##### Choix du notre client et création d' lenvoie du body pour requêter l'api

st.title('Projet 7: Implémentez un modèle de scoring')


null=np.nan

col11,col12=st.columns(2)

id_bk = col12.selectbox("SK_ID_CURR", pd.unique(df_client.index))
col11.text(f"  \n  Choisir index correspondant au client souhaité \n Votre choix:\n                     {id_bk}")


df_client[df_client.select_dtypes('object').columns]=df_client.select_dtypes('object').fillna('Autre')
df_client[df_client.select_dtypes(exclude='object').columns]=df_client.select_dtypes(exclude='object').fillna(0)
dic_id=df_client.loc[id_bk].to_dict()
our_client=pd.DataFrame(df_client.loc[id_bk].to_dict(),index=[20])
st.markdown("""---""")



###### requete à l'api


res=requests.post('https://p7mbapi.herokuapp.com/predict_val_shap_neigh',json=dic_id).json()


##### Transformation du body de retour de l'API

pred_class=res["pred_class"]
pred_proba=res["pred_proba"]

shap_val=pd.DataFrame(res["shap_val"],index=['Shap locale']).append(df_shap)



Nearest_client=pd.DataFrame(res["Nearest_client"]).reset_index(drop=True)
#Nearest_client[Nearest_client.select_dtypes('object').columns]=Nearest_client.select_dtypes('object').fillna('Autre')
#Nearest_client[Nearest_client.select_dtypes(exclude='object').columns]=Nearest_client.select_dtypes(exclude='object').fillna(0)

#Nearest_client
Nearest_client['NearestNeighbors']='Nearest Neighbors'


def res_pred_thresh(pred,predprob):
    if pred==0:
        if (1-predprob)>=0.49:
            return 1,(1-predprob)
        else: return 0, (1-predprob)
    else:
        return pred,predprob
		
def int_to_sent(p,pb):
    if p==0:
        return 'est prédit comme bon client avec une probabilité de ' +str(np.round((1-pb)*100,1))+'%'
    else:
        return 'est prédit comme un client allant faire défault avec une probabilité de ' +str(np.round((pb)*100,1))+'%'

def retrieve_col(col_,col=['CLOSED','ACTIVE','BURO','APPROVED','REFUSED','PREV','POS','INSTAL','CC'],des=df_col):
    if col_ in des.Row.values:
        return des.set_index('Row').loc[col_].to_dict()
    else: 
        lst_col=col_.split('_')
        for i in range(len(lst_col)):
            if '_'.join(lst_col[:-i]) in des.Row.values:
                d_res=des.set_index('Row').loc['_'.join(lst_col[:-i])].to_dict()
                d_res['info suppl']=', '.join(lst_col[-i:])
                return d_res

        if lst_col[0] in col:
            lst_col2=lst_col[1:]
            for i in range(len(lst_col2)):

                if '_'.join(lst_col2[:-i]) in des.Row.values:

                    d_res=des.set_index('Row').loc['_'.join(lst_col2[:-i])].to_dict()
                    d_res['info suppl']=', '.join([lst_col[0]]+lst_col2[-i:])
                    return d_res
    return {'info suppl':"Voir la référence dans le notebook d'eda et de modélisation utilisé comme exemple https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script"}
        



###  affichage des résultats

col21a,col21b,col22=st.columns([1,1,2])


pr,pb=res_pred_thresh(pred_class,pred_proba)


our_client['TARGET']=pr
our_client['NearestNeighbors']='Selected client'

all_df_fig=Nearest_client.append(our_client)

if pr==1:
    shap_val.loc['Shap locale']=-1*(shap_val.loc['Shap locale'])


labels=['Le client rembourse','Le client de remboursera pas']
sizes=[1-pb,pb]



fig, ax = plt.subplots(figsize=(3,3))
ax.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=45,explode=[0.1,0])
ax.set_title('Prédiction Remboursement crédit',fontsize=14)
#ax.legend(fontsize = 5)

ax.axis('equal')
col22.pyplot(fig)

st.markdown("""---""")








##### 6 infos sur le profil




col21a.metric(
    label=' '.join('CODE_GENDER'.lower().split('_')).capitalize(),
    value='Femme' if our_client['CODE_GENDER'].values[0]=="F" else 'Homme' )
col21a.markdown("""---""")


col21b.metric(
    label=' '.join('NAME_CONTRACT_TYPE'.lower().split('_')).capitalize(),
    value=our_client['NAME_CONTRACT_TYPE'].values[0])
col21b.markdown("""---""")

col21b.metric(
    label='Resultat:',
    value= 'Repaid' if pr==0 else 'Not repaid',
    delta=f"{int((1-pb)*100)}%" if pr==0 else f"{int((pb)*100)}%")

col21a.metric(
    label='age'.capitalize(),
    value=int(-1*our_client['DAYS_BIRTH'].values[0]/365))





def fig_by_type(d_,col_,chx_hue=choice_hue):
    limy=0
    fig, ax = plt.subplots(figsize=(6,4))
    if (str(d_[col_].dtypes)=='float64') or ((str(d_[col_].dtypes)=='int64') and d_[col_].nunique()>=3):
        if choice_plot=='Histogramme':
            ax=sns.histplot(x=col_,hue=chx_hue,data=d_,bins=20)
        else:
            ax=sns.kdeplot(x=col_,hue=chx_hue,data=d_)
        limy,chx_hue=ax.get_ylim(),choice_hue
        ax.vlines(d_.loc[20,col_],0,limy[1],label='Selected client',colors='red',linestyles='dashed')
        #ax.legend(fontsize=5)
        ax.tick_params(labelsize=8)
        ax.set_title(col_+': '+str(np.round(d_.loc[20,col_],4)) ,fontsize=10)
        ax.grid()
        ax.set_ylabel('Density',fontsize=10)

        ax.set_xlabel('')
        return fig,ax

    if ((str(d_[col_].dtypes)=='object') or d_[col_].nunique()<3):
        ax = sns.countplot(y=col_, hue=chx_hue, data=d_)

        ax.legend(fontsize=9)
        ax.tick_params(labelsize=8,rotation=25)
        ax.grid()
        ax.set_ylabel('')
        ax.set_xlabel('Count',fontsize=10)

        ax.set_title(col_+': '+str(d_.loc[20,col_]),fontsize=12)
        return fig,ax
    
##### Affichage les graphs des variables







st.header('4 graphs features importance prédiction: '+choice_db)

all_df_fig.TARGET=all_df_fig.TARGET.replace({1:'Défaut',0:'Pas de défaut'})








if choice_db=='Locale':
    shapcol=shap_val.T['Shap locale'].sort_values(ascending=False).iloc[:4].index


    col31,col32=st.columns(2)


    fig,ax=fig_by_type(all_df_fig[['TARGET','NearestNeighbors']+list(shapcol)],shapcol[0])
    col31.pyplot(fig)
    exp31=col31.expander("INFO: "+shapcol[0])
    rtc=retrieve_col(shapcol[0],des=df_col)
    for k in retrieve_col(shapcol[0],des=df_col):
        exp31.write(k+' :   '+str(rtc[k]))
 


    fig,ax=fig_by_type(all_df_fig[['TARGET','NearestNeighbors']+list(shapcol)],shapcol[1])
    col32.pyplot(fig)
    exp32=col32.expander("INFO: "+shapcol[1])
    rtc=retrieve_col(shapcol[1],des=df_col)
    for k in retrieve_col(shapcol[1],des=df_col):
        exp32.write(k+' :   '+str(rtc[k]))

    col41,col42=st.columns(2)  


    fig,ax=fig_by_type(all_df_fig[['TARGET','NearestNeighbors']+list(shapcol)],shapcol[2])
    col41.pyplot(fig)
    exp41=col41.expander("INFO: "+shapcol[2])
    rtc=retrieve_col(shapcol[2],des=df_col)
    for k in retrieve_col(shapcol[2],des=df_col):
        exp41.write(k+' :   '+str(rtc[k]))


    fig,ax=fig_by_type(all_df_fig[['TARGET','NearestNeighbors']+list(shapcol)],shapcol[3])
    col42.pyplot(fig)
    exp42=col42.expander("INFO: "+shapcol[3])
    rtc=retrieve_col(shapcol[3],des=df_col)
    for k in retrieve_col(shapcol[3],des=df_col):
        exp42.write(k+' :   '+str(rtc[k]))

if choice_db=='Globale':
    shapcol=shap_val.T['Shapley_importance'].sort_values(ascending=False).iloc[:4].index
    col31,col32=st.columns(2)


    fig,ax=fig_by_type(all_df_fig[['TARGET','NearestNeighbors']+list(shapcol)],shapcol[0])
    col31.pyplot(fig)
    exp31=col31.expander("INFO: "+shapcol[0])
    rtc=retrieve_col(shapcol[0],des=df_col)
    for k in retrieve_col(shapcol[0],des=df_col):
        exp31.write(k+' :   '+str(rtc[k]))
 


    fig,ax=fig_by_type(all_df_fig[['TARGET','NearestNeighbors']+list(shapcol)],shapcol[1])
    col32.pyplot(fig)
    exp32=col32.expander("INFO: "+shapcol[1])
    rtc=retrieve_col(shapcol[1],des=df_col)
    for k in retrieve_col(shapcol[1],des=df_col):
        exp32.write(k+' :   '+str(rtc[k]))

    col41,col42=st.columns(2)  


    fig,ax=fig_by_type(all_df_fig[['TARGET','NearestNeighbors']+list(shapcol)],shapcol[2])
    col41.pyplot(fig)
    exp41=col41.expander("INFO: "+shapcol[2])
    rtc=retrieve_col(shapcol[2],des=df_col)
    for k in retrieve_col(shapcol[2],des=df_col):
        exp41.write(k+' :   '+str(rtc[k]))


    fig,ax=fig_by_type(all_df_fig[['TARGET','NearestNeighbors']+list(shapcol)],shapcol[3])
    col42.pyplot(fig)
    exp42=col42.expander("INFO: "+shapcol[3])
    rtc=retrieve_col(shapcol[3],des=df_col)
    for k in retrieve_col(shapcol[3],des=df_col):
        exp42.write(k+' :   '+str(rtc[k]))



st.markdown("""---""")

st.header('Features importance: Local vs Global')


col_shaploc=abs(shap_val.T['Shap locale']).T.sort_values(ascending=False).iloc[:10].index.values
dffig=(shap_val.T['Shap locale'].loc[col_shaploc].reset_index())

dffig['PI']=dffig['Shap locale']>0


fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(16,5))

sns.barplot(y='Shapley_importance', x='Features', data=df_shap.T['Shapley_importance'].sort_values(ascending=False).iloc[:10].reset_index(),ax=axes[1],color='blue')
axes[1].tick_params(labelsize=10,rotation=90)
axes[1].grid()
axes[1].set_ylabel('Shapley Values: Importance',fontsize=10)
axes[1].set_xlabel('',fontsize=10)
axes[1].set_title('Global importance',fontsize=10)

axes[1].grid()


sns.barplot(y='Shap locale', x='index', data=dffig,hue='PI',ax=axes[0])
axes[0].set_ylabel('Shapley Values: Importance',fontsize=10)

axes[0].tick_params(labelsize=10,rotation=90)
axes[0].grid()
axes[0].grid()
axes[0].set_xlabel('',fontsize=10)
axes[0].set_title('Local importance',fontsize=10)

axes[0].grid()

st.pyplot(fig)

