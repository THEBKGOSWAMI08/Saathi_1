import streamlit as st
import json
import os

# Custom CSS styling for a beautiful interface
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    
    .main-heading {
        font-size: 45px;
        color: #2a9d8f;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
    }

    .subheading {
        font-size: 25px;
        color: #264653;
        margin-top: 30px;
        text-align: center;
    }

    .task-container {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
        border: 1px solid #ced4da;
    }

    .task-input {
        font-size: 18px;
        padding: 10px;
        border: none;
        border-radius: 8px;
        background-color: #ffffff;
        width: 100%;
        margin-top: 10px;
    }

    .task-button {
        background-color: #2a9d8f;
        color: white;
        font-size: 18px;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        margin-top: 15px;
    }

    .task-button:hover {
        background-color: #21867a;
    }

    .task-item {
        font-size: 20px;
        padding: 10px;
        background-color: #ffffff;
        margin-top: 10px;
        border: 1px solid #ced4da;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .remove-button {
        background-color: #e76f51;
        color: white;
        border: none;
        padding: 5px 15px;
        border-radius: 5px;
        cursor: pointer;
    }

    .remove-button:hover {
        background-color: #c35441;
    }

    .completed-task {
        text-decoration: line-through;
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# File to store tasks
TASK_FILE = "tasks.json"

# Function to load tasks from the JSON file
def load_tasks():
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE, "r") as f:
            return json.load(f)
    return {"tasks": [], "completed": []}

# Function to save tasks to the JSON file
def save_tasks(tasks, completed):
    with open(TASK_FILE, "w") as f:
        json.dump({"tasks": tasks, "completed": completed}, f)

def run_todo():
    # Main heading
    st.markdown("<h1 class='main-heading'>To-Do List📝</h1>", unsafe_allow_html=True)

    # Load tasks from file or initialize session state
    if 'tasks' not in st.session_state:
        data = load_tasks()
        st.session_state['tasks'] = data['tasks']
        st.session_state['completed'] = data['completed']

    # Input task from the user
    st.markdown("<h3 class='subheading'>Add a new task:</h3>", unsafe_allow_html=True)
    new_task = st.text_input("", key='task_input', placeholder="Enter your task here...", label_visibility="collapsed")

    # Function to add task to the list and save to file
    def add_task():
        if new_task:
            st.session_state['tasks'].append(new_task)
            st.session_state['completed'].append(False)
            save_tasks(st.session_state['tasks'], st.session_state['completed'])  # Save tasks to file
            st.session_state['task_input'] = ""  # Clear input field

    # Function to remove task from the list and save to file
    def remove_task(index):
        del st.session_state['tasks'][index]
        del st.session_state['completed'][index]
        save_tasks(st.session_state['tasks'], st.session_state['completed'])  # Save updated tasks to file

    # Button to add task
    st.button("Add Task", on_click=add_task, key='add_button', use_container_width=True)

    # Display the tasks with checkboxes for completion
    st.markdown("<h3 class='subheading'>Your To-Do List:</h3>", unsafe_allow_html=True)
    if st.session_state['tasks']:
        for i, task in enumerate(st.session_state['tasks']):
            col1, col2 = st.columns([0.1, 0.9])

            # Checkbox to mark task as complete
            with col1:
                completed = st.checkbox("", key=f"check_{i}", value=st.session_state['completed'][i])
                st.session_state['completed'][i] = completed
                save_tasks(st.session_state['tasks'], st.session_state['completed'])  # Save updated completion status

            # Task description with line-through if completed
            with col2:
                task_style = "completed-task" if st.session_state['completed'][i] else ""
                st.markdown(f"<div class='task-item'><span class='{task_style}'>{task}</span>", unsafe_allow_html=True)
                st.button("Remove", key=f"remove_{i}", on_click=remove_task, args=(i,), use_container_width=True)
    else:
        st.write("You have no tasks. Add a task to get started!")

if __name__ == '__main__':
    run_todo()
