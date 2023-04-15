#imports for our app
import streamlit as st
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import pandas as pd

import calendar
from datetime import datetime

import database as db 

df = pd.read_csv(r'C:/Users/LENOVO/Documents/Hackerstellar_StrawHats/spending.csv')
#variables
# incomes = ["Salary","Blog","Other Income"]
# expenses = ["amount"]
# category=["category"]
# description=["description"]
currency = "Rs"
page_title = "Expense Stream"
page_icon = ":money_with_wings:"
layout = "centered"

#setting title for our app
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

today = datetime.today()

def get_all_periods():
    items = db.fetch_all_periods()
    periods = [item["key"] for item in items]
    return periods

hide_st_style = """
<style>
#MainMenu {visiblility: hidden;}
footer {visiblility: hidden;}
header {visiblility: hidden;}
</style>
"""

st.markdown(hide_st_style,unsafe_allow_html=True)

selected = option_menu(
    menu_title= None,
    options=["Data Entry","Data Visualization"],
    icons=["pencil-fill","bar-chart-fill"],
    orientation= "horizontal",
)


if selected == "Data Entry":
        st.header(f"Data Entry in {currency}")
        with st.form("Entry_form", clear_on_submit=True):

            # with st.expander("Income"):
            #     for income in incomes:
            #         st.number_input(f"{income}:",min_value=0, format="%i", step=10,key=income)
            # with st.expander("Expenses"):
            #     for expense in expenses:
            #         st.number_input(f"{expense}:", min_value=0,format="%i",step=10,key=expense)
            # with st.expander("Category"):
            #     Category = st.text_area("", placeholder="Enter Category hee ...")
            # with st.expander("Description"):
            #     Description = st.text_area("", placeholder="Enter Description hee ...")

            date=st.text_input("date")
            category=st.text_input("category")
            amount=st.text_input("amount")
            description=st.text_input("description")



            submitted = st.form_submit_button("Save Data")
            if submitted:
                st.write(date,category,amount,description)
                new_data = {"date": date, "category":category, "amount":amount, "description":description}
                df=df.append(new_data, ignore_index=True)
                df.to_csv(r'C:/Users/LENOVO/Documents/Hackerstellar_StrawHats/spending.csv', index=False)
                st.success("Data Saved!")





if selected == "Data Visualization":
        st.header("Data Visualization")
        with st.form("Saved periods"):
            period = st.selectbox("Select Period:", get_all_periods())
            submitted = st.form_submit_button("Plot Period")
            if submitted:
                period_data = db.get_period(period)
                for doc in period_data:
                    Category = doc["Category"]
                    expenses = doc["expenses"]
                    incomes = doc["incomes"]
                    Description = doc["Description"]


                total_income = sum(incomes.values())
                total_expense = sum(expenses.values())
                remaining_budget = total_income - total_expense
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Income",f"{total_income} {currency}")
                col2.metric("Total Expense",f"{total_expense} {currency}")
                col3.metric("Remaining Budget",f"{remaining_budget} {currency}")
                st.text(f"Category:{Category}")


                label = list(incomes.keys()) + ["Total income"] + list(expenses.keys())
                source = list(range(len(incomes))) + [len(incomes)] * len(expenses)
                target = [len(incomes)] * len(incomes) + [label.index(expense) for expense in expenses.keys()]
                value = list(incomes.values()) + list(expenses.values())


                link = dict(source=source, target=target,value=value)
                node = dict(label=label,pad=20,thickness=25,color="#00684A")
                data = go.Sankey(link=link,node=node)

                fig = go.Figure(data)
                fig.update_layout(margin=dict(l=0,r=0,t=5,b=5))
                st.plotly_chart(fig, use_container_width=True)

