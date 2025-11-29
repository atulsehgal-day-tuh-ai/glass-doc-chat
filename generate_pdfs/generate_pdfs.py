import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Create a directory to hold the files
output_dir = "generate_pdfs/sample_pdfs"
os.makedirs(output_dir, exist_ok=True)

# The 10 Leadership Topics with realistic content
documents = [
    {
        "filename": "01_Transformational_Leadership.pdf",
        "title": "Transformational Leadership Guide",
        "content": [
            "Transformational leaders inspire and motivate their workforce.",
            "They focus on the 'big picture' and strategic goals.",
            "Key trait: Emotional Intelligence (EQ) is critical for connection.",
            "Action: Encourage creativity and challenge the status quo."
        ]
    },
    {
        "filename": "02_Conflict_Resolution.pdf",
        "title": "Conflict Resolution in Management",
        "content": [
            "Conflict is inevitable in dynamic teams.",
            "The Thomas-Kilmann model identifies 5 styles: Avoiding, Accommodating,",
            "Competing, Compromising, and Collaborating.",
            "Managers should focus on 'Interest-Based Relational' approaches."
        ]
    },
    {
        "filename": "03_Strategic_Decision_Making.pdf",
        "title": "Frameworks for Strategic Decisions",
        "content": [
            "Decision making requires data, intuition, and risk assessment.",
            "SWOT Analysis: Strengths, Weaknesses, Opportunities, Threats.",
            "PESTLE Analysis: Political, Economic, Social, Tech, Legal, Environmental.",
            "Avoid 'Analysis Paralysis' by setting strict deadlines."
        ]
    },
    {
        "filename": "04_Remote_Team_Management.pdf",
        "title": "Leading Remote and Hybrid Teams",
        "content": [
            "Remote work requires over-communication, not micromanagement.",
            "Focus on outcomes (results) rather than hours worked.",
            "Use asynchronous tools (Slack, Teams) to respect time zones.",
            "Schedule regular 'virtual coffees' to maintain team bonding."
        ]
    },
    {
        "filename": "05_Agile_Leadership.pdf",
        "title": "Principles of Agile Leadership",
        "content": [
            "Agile is about iterative development and adaptability.",
            "Servant Leadership: The leader exists to serve the team.",
            "Fail fast, learn faster: Mistakes are learning opportunities.",
            "Sprints: Short, focused periods of work (usually 2 weeks)."
        ]
    },
    {
        "filename": "06_Giving_Feedback.pdf",
        "title": "The Art of Constructive Feedback",
        "content": [
            "Feedback should be specific, actionable, and timely.",
            "The SBI Model: Situation, Behavior, Impact.",
            "Avoid the 'Sandwich Method' (praise-criticism-praise) as it confuses.",
            "Focus on the behavior, not the person's character."
        ]
    },
    {
        "filename": "07_Change_Management.pdf",
        "title": "Leading Through Change",
        "content": [
            "Kotter's 8-Step Process for Leading Change.",
            "1. Create Urgency. 2. Form a Coalition. 3. Create a Vision.",
            "Resistance is natural; address fears directly.",
            "Celebrate short-term wins to build momentum."
        ]
    },
    {
        "filename": "08_Emotional_Intelligence.pdf",
        "title": "EQ in the Workplace",
        "content": [
            "Self-Awareness: Recognizing your own triggers.",
            "Self-Regulation: Controlling impulsive reactions.",
            "Empathy: Understanding the emotions of others.",
            "Social Skills: Managing relationships to move people in desired directions."
        ]
    },
    {
        "filename": "09_Delegation_Mastery.pdf",
        "title": "Effective Delegation Strategies",
        "content": [
            "Delegation is not dumping work; it is empowerment.",
            "Select the right person for the task based on skills.",
            "Clearly define the desired outcome/result.",
            "Provide necessary resources and authority, then step back."
        ]
    },
    {
        "filename": "10_Time_Management_Execs.pdf",
        "title": "Time Management for Executives",
        "content": [
            "The Eisenhower Matrix: Urgent vs. Important.",
            "Deep Work: Blocking out distraction-free time for strategy.",
            "The Pareto Principle (80/20 Rule): 20% of efforts give 80% of results.",
            "Learn to say 'No' to protect your strategic focus."
        ]
    }
]

print(f"Generating {len(documents)} PDF documents...")

for doc in documents:
    file_path = os.path.join(output_dir, doc["filename"])
    c = canvas.Canvas(file_path, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, doc["title"])
    
    # Line
    c.setStrokeColor(colors.blue)
    c.line(50, 740, 550, 740)
    
    # Content
    c.setFont("Helvetica", 12)
    y_position = 710
    for line in doc["content"]:
        c.drawString(50, y_position, f"- {line}")
        y_position -= 20
        
    c.save()
    print(f"Created: {doc['filename']}")

print(f"\n✅ Done! Check the '{output_dir}' folder.")