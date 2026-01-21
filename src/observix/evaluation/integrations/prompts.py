STANDARD_EVALUATION_TEMPLATE = """
CRITICAL INSTRUCTIONS:
- Base ALL analysis STRICTLY on actual trace data - cite specific aget names, tool names and execution patterns
- Provide evidence in structured array format with specific categories
- Give crisp, actionable reasoning focused on observed workflow behavior
- Avoid generic suggestions - only recomment improvements based on actual data patterns
- Focus on quantifiable metrics from the trace execution

EVIDENCE REQUIREMENTS: 
- Extract specific agent names that performed actions 
- Identify exact tool names used in execution
- Document actual execution patterns and sequences
- Avoid mentioning trace ids, metadata or technical implementation details
"""

TOOL_SELECTION_PROMPT = """
{standard_evaluation}

TOOL SELECTION RUBRIC (1-100 Scale):
{rubric_score_guidelines}
Specific Criteria:

Score 90-100 (Exceptional):
•⁠ 95-100% of tool selections are optimal and directly relevant
• Perfect mapping between task requirements and tool capabilities
•⁠ No missed critical tools, zero unnecessary selections
•⁠ Tools enhance workflow efficiency by 40%+ compared to alternatives

Score 70-80 (Proficient): I
• 85-94% of tool selections are appropriate and effective
• Good understanding of tool-task alignment with 1-2 minor suboptimal choices
•⁠ Minimal missed opportunities (1-2 instances)
•⁠ Tools contribute positively to workflow with slight efficiency gains.

Score 50-60 (Adequate):
•⁠ 70-84% of tool selections meet basic requirements
•⁠ Adequate tool-task matching with 3-4 questionable choices
•⁠ Some missed optimization opportunities but core functionality maintained
•⁠ Tools fulfill basic needs without significant efficiency impact

Score 30-40 (Below Standard):
•⁠ 50-69% of tool selections are appropriate
•⁠ Poor tool-task alignment with multiple inappropriate choices
•⁠ Several critical tools missed, causing workflow inefficiencies
•⁠ Tool selection creates obstacles rather than solutions

Score 10-20 (Unacceptable):
•⁠ Less than 50% of tool selections are appropriate
•⁠ Fundamental misunderstanding of tool purposes
•⁠ Critical workflow failures due to wrong tool choices
•⁠ Tool selection actively prevents task completion



"""

TOOL_INPUT_STRUCTURE_PROMPT_TEMPLATE = """
You are an expert evaluation assistant evaluating the input structure of the input 
whether they are perfectly formatted and schema-compliant. Do they have malformed inputs, 
does the input maximize tool potential and enable optimal performance?

    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Tool Called]: {tool_call}
    ************
    [Trace]: {trace}
    [END DATA]

Example Scoring Rubric
0: The input is incorrect
0.5: The input is partially correct
1: The input is correct

#QUESTION
Please rate the percentage of correctness in the context on a scale from 0 to 1.
"""

TOOL_SEQUENCE_PROMPT_TEMPLATE = """
You are an expert evaluation assistant evaluating the tool sequence of the tool calls   

    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Tool Sequence]: {tool_sequence}
    ************
    [Trace]: {trace}
    [END DATA]

Example Scoring Rubric
0: The tool sequence is incorrect
0.5: The tool sequence is partially correct
1: The tool sequence is correct

[Tool Definitions]: {tool_definitions}

#QUESTION
Please rate the percentage of correctness in the context on a scale from 0 to 1.
"""

AGENT_ROUTING_PROMPT_TEMPLATE = """
You are an expert evaluation assistant evaluating the routing decisions, agent selection and 
workflow coordination based on the actual agents and tools used in this specific trace.

    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Trace]: {trace}
    [END DATA]

Example Scoring Rubric
0: The routing decisions are incorrect
0.5: The routing decisions are partially correct
1: The routing decisions are correct

[Agent Definitions]: {agent_definitions}

#QUESTION
Please rate the percentage of correctness in the context on a scale from 0 to 1.
"""

HITL_PROMPT_TEMPLATE = """
You are an expert evaluation assistant evaluating the Human-in-the-loop (HITL) decisions 

    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Trace]: {trace}
    [END DATA]

Example Scoring Rubric
0: The HITL decisions are incorrect
0.5: The HITL decisions are partially correct
1: The HITL decisions are correct

[HITL_INFO]: {HITL_INFO}
[Agent Definitions]: {agent_definitions}

#QUESTION
Please rate the percentage of correctness in the context on a scale from 0 to 1.
"""

WORKFLOW_COMPLETION_PROMPT_TEMPLATE = """
You are an expert evaluation assistant evaluating the workflow completion, how effectively the workflow has completed the intended task

    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Trace]: {trace}
    [END DATA]

Example Scoring Rubric
0: The workflow completion is incorrect
0.5: The workflow completion is partially correct
1: The workflow completion is correct

[Agent Definitions]: {agent_definitions}

#QUESTION
Please rate the percentage of correctness in the context on a scale from 0 to 1.
"""

CUSTOM_METRIC_PROMPT_TEMPLATE = """
You are an expert evaluation assistant evaluating the custom metric, how effectively the custom metric has completed the intended task

    [BEGIN DATA]
    ************
    [Question]: {question}
    ************
    [Trace]: {trace}
    [END DATA]

Example Scoring Rubric
0: The custom metric is incorrect
0.5: The custom metric is partially correct
1: The custom metric is correct

[Agent Definitions]: {agent_definitions}
[Custom Instructions]: {custom_instructions}
#QUESTION
Please rate the percentage of correctness in the context on a scale from 0 to 1.    
"""

TRACE_SANITIZATION_PROMPT = """
You are sanitizing trace data from a multi-agent system. Your goal is to extract ONLY agent execution steps. 
Workflow Structure: 
Agents -> {agents}
Tools -> {tools}

AGENT COUNT VALIDATION REQUIREMENT:
**Before proceeding with sanitization, extract the expected agent count from the workflow structure.
After sanitization is complete, verify that the number of unique agents in the sanitized output matches the expected count.
This prevents unintended agent filtering or data loss during processing.

Validation Steps:
    1. Count expected agents from workflow structure
    2. Perform sanitization following the rules below
    3. Count unique agents in sanitized output (by unique agent_name/node_name)
    4. If counts don't match, review filtering rules to ensure no valid agents were excluded
    5. Include validation metadata in output: {{"expected_agent_count": X, "sanitized_agent_count": Y, "validation_passed": true/false}}
    6. Don't consider tool execution as agent and ensure in the sanitized output there are no tool calling and LLM calling as agents.

**CRITICAL AGENT IDENTIFICATION RULES:**
1. An observation is an AGENT STEP if and ONLY if: it has a `node.type=='agent'` - then only it will represent an actual agent decision / action, not internal LLM processing.
2. NEVER treat these as agent steps: 
    - Any observation with `name` having ['invoke'] is not an agent
    - Any observation with `node.type`=='tool'
3. For each valid agent step, extract the following
    - agent name
    - agent's goal
    - agent's input
    - agent's output
    - agent's timestamp
4. Maintain the chronological order using the timestamps
5. **POST-SANITIZATION CHECK:**
    Verify that all agents mentioned in the `Workflow Structure` appear in the sanitized output, if an agent is missing, flag it as a potential data integrity issue rather than silently excluding it.

** OUTPUT FORMAT (STRICT JSON) **
{{
"sanitized_trace": [
    {{
        "agent": "RoutingAgent", 
        "goal": "Route user request to appropriate handler", 
        "input": {{"user_query": "..."}}
        "output": {{"next_agent": "..."}}
        "timestamp": "2024-01-15T10:30:00"
    }}
]
}}

**Validation**
- Every entry MUST have a valid agent name 
- Eliminate duplicate entries for the same agent action 
- Keep only semantically meaningfull agent steps 

Raw Trace Data: {trace}
"""