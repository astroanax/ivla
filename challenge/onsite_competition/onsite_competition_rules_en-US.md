# Manip Track Onsite Competition Rules
English Version | [中文版](./onsite_competition_rules_zh-CN.md)

## 1. Task Description
This track aims to develop a multimodal robotic operating system with language understanding and execution capabilities. Participants are required to design an end-to-end control policy model that completes the full pipeline from visual perception and natural language instruction understanding to action control prediction. Robots will operate in an open tabletop environment, controlling robotic arms to complete various manipulation tasks under complex natural language instructions.  

Main challenges include:
- Integrating language and visual information to drive a unified perception–decision–control pipeline
- Handling long-horizon tasks that challenge model stability and self-correction capabilities in decision-making and control
- Coping with diverse scene layouts, object types, and task instructions, which challenge model generalization

## 2. Competition Environment and Equipment
To ensure fairness and reproducibility, the organizing committee will provide standardized hardware and venue setups, including but not limited to:  

**Competition venue**:  
A standardized tabletop workspace (recommended size approximately 1.5 m × 1.0 m). The final size and boundary calibration will follow the official layout drawings published by the organizers.

**Objects**:  
Common household, educational, and daily-use items. All items will be uniformly labeled and managed, ensuring identical models, sizes, materials, and colors for all teams.  

**Standardization and fairness**:
- All teams will face the exact same set of tasks (around five groups, each containing two test cases), and all cases will include natural language task instructions to be executed in the same order.
- The initial tabletop state (including layout, object positions, orientations, poses, etc.) for each case will be pre-arranged and calibrated by referees to ensure environment consistency.

**Robot platform**:  
The unified robot platform provided by the organizers and its paired robotic arm.   

**Sensor system**:  
A standardized RGB-D camera with provided extrinsic/intrinsic calibration parameters. The mounting position, viewing angle, and frame rate of the sensor will also be standardized and published in an official notice prior to the competition.

## 3. Task Setup
**Task format**:  
All task instructions will be given in natural language text, covering typical manipulation modes such as pick, place, stack, and insert. The test set will be sourced from unseen data to evaluate model generalization to novel environments.

**Key sub-goals**:  
Some complex tasks are broken down into multiple key sub-goals.

For example, the task of "Putting the colored blocks on the table into the corresponding colored bowls" can be broken down into:
1. Place the red blocks in the red bowl;
2. Place the yellow blocks in the yellow bowl; 

**Evaluation rules**:  
 The success criteria and detection logic for each sub-goal will be standardized by the referee team and published as a technical manual before the competition. This ensures all teams are evaluated under the same rules, improving repeatability, fairness, and transparency.

## 4. Competition Procedure
### 4.1 Pre-competition Preparation
- Teams must package the competition image in advance according to the GitHub documentation.
- A standardized debugging time and environment will be provided onsite. Teams may use model weights different from the online stage and make environment-specific configuration adjustments, but modifications to the core algorithm logic are strictly prohibited.

### 4.2 Task Execution
- Each team will execute all pre-set tasks in sequence. Before each task starts, the scene will be reset to the official standard initial state by the referees.
- Execution flow: team confirms system startup → referee inputs natural language instruction  → raise hand to signal → referee starts timing → system executes automatically.
- No human intervention related to the algorithm is allowed during execution. In the event of system freeze/crash, a one-time restart may be allowed with referee approval.

### 4.3 Time Limits
- Each task has a maximum allowed duration 5 minutes (to be announced by the organizers based on pre-competition tests).
- Each team’s total competition time (including switching, preparation, and execution) must not exceed 55 minutes.

### 4.4 Retry and Abandonment
- If dropping, freezing, or abnormal poses occur during execution, teams may choose to:
  - Retry the current task (maximum 2 times; time is counted continuously), or
  - Abandon the current task (score = 0, proceed to next task)
- If time exceeds the maximum limit, referees may forcibly stop the current task and record the result.

### 4.5 Fairness and Compliance
- Collecting additional data on site is prohibited. Using private datasets for offline or collaborative pre-training before the competition is allowed, but using on-site images to update the model is forbidden.  
- Except for fixing system freeze/crash, modifying core algorithm code during the competition is strictly forbidden.
- Any unauthorized human intervention (including but not limited to manually moving objects, adjusting robot arm poses, modifying task judgment logic, etc.) is prohibited. Violations will result in point deductions or disqualification.
- Teams engaging in cheating or disorderly behavior may be disqualified by the organizing committee.

## 5. Scoring Rules
### 5.1 Scoring Formula (Onsite Competition)
- Score per task = (Number of completed sub-goals / Total sub-goals) × 100%
- Team total score = Avg of all task scores

### 5.2 Final Results
The final result is calculated by weighting the scores of the online and offline stages:  
- **Final Score Calculation**:  
Final Score = (Online Points × 40%) + (Onsite Points × 60%)  

## 6. Supplementary Provisions
- Referee assignment: At least 2 referees per match, responsible for timing, scoring, recording execution, and judging behaviors. Referees must remain neutral and may not assist any team.
- Fair competition: Teams must follow academic integrity and competition ethics. Plagiarism, cheating, and tampering with system logs are strictly prohibited.
- Information disclosure: The committee will publish the evaluation protocols, scoring logic, device list, initial layout drawings, and interface documentation in advance to ensure all teams compete on the same information baseline.
- Right to revise rules: In case of force majeure or emergencies (e.g., equipment failure, site unavailability, safety risks), the organizing committee reserves the right to make minor adjustments to the rules and will notify all teams in advance through official channels.
