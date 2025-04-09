- Restructure files and analysis for easier access, change permissions for certain set of users
  {View access for docs outside organization drive folder is still buggy}
  
- Find the cause for variability in Visit 5 {Doing Root Cause Analysis on 3 factors}
- Sensor Validation {Aggregation only for given time periods} :
    Implemented but has errors; fixed one part working on the other 
- 3T visuals fix Non Contingent breaks in graph {false face detection rows}
- Change Toy Status Value Counts {Validation Page} : 3T Visuals
- Better Bin Visualization 
- Remove bars based on clock time {Prone Time}

- Refine Pandas Agent {Solve the biggest bugs first} : Need to preprocess files !  
  1. Need better few shot examples
  2. Auto-evals {Human based evals are taking long}/ LLM as a judge
  3. Vectorize the metadata
  4. First check --> Syntax errors, Second check --> Logical Error
 
  Try :
  Routing {Based on the context to the correct dataset}
  Synthetic data generation ? 
