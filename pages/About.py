import streamlit as st

def about_page():
    st.title("About 😎 Syed Mustafa Ali")

    
    st.header("🤓 Introduction")
    st.write("I am a Team Leader with over 7 years of experience, leading 30+ employees processing US Healthcare claims. "
             "My expertise includes Quality Analysis, Health Insurance, US Healthcare, and Claims Processing. I am skilled in Python, JavaScript, HTML, and CSS. I have successfully improved quality and saved resources in 6 projects while meeting SLAs.")

    # Work History
    st.header("😁 Work History")

    # You can format your work history as a list or table here.
    work_history = [
        {
            "Period": "April 2023 - Current",
            "Position": "Team Leader",
            "Company": "Carelon Global Solutions, Hyderabad",
            "Description": "Team Leader for a healthcare project at Carelon Global Solutions. Responsible for managing a team and ensuring productivity and quality goals are met."
        },
        {
            "Period": "2021-01 - 2023-04",
            "Position": "Team Lead",
            "Company": "Wipro, Hyderabad",
            "Description": "Team Lead for a healthcare project in Wipro (Anthem Inc - GBD Claims). Managed a team of 30+ members. Conducted daily reviews of productivity and quality scores for quicker issue identification. Prepared performance reports."
        },
        {
            "Period": "2020-08 - 2021-01",
            "Position": "Trainer",
            "Company": "Wipro, Hyderabad",
            "Description": "Mentored new hires, scheduled and taught in-class and online courses, monitored participant workflow, and performed continuous evaluations of content."
        },
        # Add other work history entries here
    ]

    for entry in work_history:
        st.subheader(f"{entry['Period']} - {entry['Position']}")
        st.write(f"Company: {entry['Company']}")
        st.write(f"Description: {entry['Description']}")

    # Skills
    st.header("😎 Skills")
    
    # Format your skills list here.
    skills = [
        "Team Handling",
        "VBA",
        "HTML",
        "CSS",
        "Python",
        "JavaScript",
        "Quality Control",
        "Technical Support",
        "MS Office Advance",
        "Six Sigma",
        "Managerial Skills",
        "Quality Improvement"
    ]
    
    st.write(", ".join(skills))

    # Software
    st.header("🧑‍💻 Software")
    
    # Format your software skills here.
    software_skills = [
        "Python",
        "MS VBA",
        "MS Excel",
        "MS Word",
        "MS PowerPoint",
        "Selenium",
        "Tkinter",
        "Facets",
        "Macess",
        "Claims Workstation",
        "Tally",
        "Power Automate",
        "Node JS",
        "Express JS",
        "SQL",
        "Mangoose",
        "EJS",
        "MICROFOCUS VBA."
        
    ]
    
    st.write(", ".join(software_skills))

    # Projects
    st.header("😁 Projects")
    
    # Format your project list here.
    projects = [
        "Corrected Claim Auto Keyer BOT - A Bot that adds changes from Corrected Image to Facets Original claim.",
        "Cob Auto Keyer - A python based auto keyer that adds EOB data from Macess Image to Facets.",
        "CWS Corrected Auto Keyer- A python based Bot that Completes checks all the criteria as per the PI CWS Corrected claims end to end.",
        "COB Auto adjudicator end to end COB claims Processing BOT.",
        "Data Elements Checker BOT",
        "Segregation BOT it Checks type of claim EDI, paper, COB, Authorization or Corrected."
    ]
    
    for project in projects:
        st.write(f"- {project}")

    # Personal Information
    st.header("🫡 Personal Information")
    st.write("✅ Email: syedmustafa177@gmail.com")
    st.write("✅ LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/syedmustafa177)")
    st.write("✅ GitHub: [GitHub Profile](https://github.com/Syedmustafa177)")
    st.write("✅ Website: [Personal Website](https://bold.pro/my/syed-mustafa-230301164919/757)")
    st.write("✅ Twitter: [@SyedMustafa177](https://twitter.com/SyedMustafa177)")

    # Introduction

if __name__ == "__main__":
    about_page()
