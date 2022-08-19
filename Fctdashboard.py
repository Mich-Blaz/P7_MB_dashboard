# -*- coding: utf-8 -*-
"""


@author: MichBlaz

Description: Les différentes fonctions qui seront utilisées dans le dashboard

"""

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