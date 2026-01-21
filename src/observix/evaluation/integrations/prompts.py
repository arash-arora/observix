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

AVAILABLE WORKFLOW RESOURCES:
Available Agents: {agents}
Available Tools: {tools}
Use ONLY these available resources for evaluation. Do not suggest tools or agents not present in this context.
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

ANALYSIS FOCUS:
- Tool-task alignment: Match between chosen tools and actual task requirements
- Selection efficiency: Optimal choices vs missed opportunities
- Task completion impact: How tool choices affected outcome quality
- Resource utilization: Use of available tools relative to workflow needs

Provide response in JSON format:
{{
"tool_selection": {{
    "score": <int 10-100>,
    "reasoning": "<Crisp analysis: Which tools were used effectively, which were suboptimal, and why based on actual execution patterns>",
    "evidences": [
        {{"type": "tools_used", "description": "Primary tools utilized", "items": ["<tool_name_1>", "<tool_name_2>"]}},
        {{"type": "agents_involved", "description": "Agents that performed tool selection", "items": ["<agent_name_1>", "<agent_name_2>"]}},
        {{"type": "selection_patterns", "description": "Observed tool selection behaviors", "details": ["<pattern_1: agent_x consistently chose tool_y for task_z>", "<pattern_2>"]}},
        {{"type": "effectiveness_examples", "description": "Specific tool usage outcomes", "examples": ["<example_1: specific tool use and its outcome>", "<example_2>"]}}
    ],
    "performance_metrics": {{
        "total_tool_calls": <int>,
        "optimal_selections": <int>,
        "suboptimal_selections": <int>,
        "effectiveness_rate": <float>
    }},
    "feedback": [
        {{"category": "tool_optimization", "recommendation": "<Crisp feedback 1: specific improvement based on observed tool usage>", "examples": "<Example 1: provide examples similar to the context which could get higher scores>"}},
        {{"category": "workflow_enhancement", "recommendation": "<Crisp feedback 2: actionable recommendation from execution patterns>", "examples": "<Example 1: provide examples similar to the context which could get higher scores>"}}
    ]
}}
}}

Sanitized Trace Data: {trace_data}
"""

TOOL_INPUT_STRUCTURE_PROMPT_TEMPLATE = """
{standard_evaluation}

INPUT STRUCTURE RUBRIC (10-100 Scale):
{rubric_score_guidelines}
Specific Criteria:

Score 90-100 (Exceptional):
•⁠ 95-100% of inputs are perfectly formatted and schema-compliant
•⁠ Comprehensive parameter utilization with advanced configurations
•⁠ Zero malformed inputs, complete error handling implemented
•⁠ Inputs maximize tool potential and enable optimal performance

Score 70-80 (Proficient):
•⁠ 85-94% of inputs are well-structured and compliant
•⁠ Good parameter usage with appropriate detail level
•⁠ Minor formatting inconsistencies (1-2 instances)
• Inputs effectively leverage most tool capabilities

Score 50-60 (Adequate):
•⁠ 70-84% of inputs meet basic structural requirements
•⁠ Adequate formatting with some inconsistencies
•⁠ Basic parameter usage, some capabilities underutilized
•⁠ Functional inputs but with room for optimization

Score 30-40 (Below Standard):
•⁠ 50-69% of inputs are properly structured
•⁠ Frequent formatting errors and schema violations
⁠•⁠ Insufficient detail causing suboptimal tool execution
•⁠ Many capabilities unused due to poor input construction

Score 10-20 (Unacceptable):
•⁠ Less than 50% of inputs are properly formatted
•⁠ Widespread schema violations and malformed data
•⁠ Critical parameters missing, causing tool failures
•⁠ Inputs prevent successful tool execution

ANALYSIS FOCUS:
- Parameter completeness: Required fields present and properly formatted
- Data quality: Clean, relevant input data without noise or missing values
- Structure compliance: Adherence to expected tool input schemas and data types
- Utilization effectiveness: How well inputs leveraged tool capabilities and avoided redundant data

**EVALUATION REQUIREMENTS:**

1. **Trace-Based Analysis**: Reference SPECIFIC tool invocations with actual input parameters from trace_data
2. **Schema Validation**: Compare actual inputs against expected tool schemas from eval_context_info
3. **Quality Patterns**: Identify concrete formatting successes and failures with examples
4. **Quantifiable Metrics**: Measure completeness rates, missing fields, and type mismatches

Provide response in JSON format:
{{
    "input_structures": {{
        "score": <int 10-100>,
        "reasoning": "<State SPECIFIC input quality patterns observed. Format: 'X/Y tool invocations had complete parameters. <ToolName1> received well-structured [field_name] in [data_type] format. <ToolName2> had missing [required_field] in [N instances], causing [observable_issue]'. Reference actual tool names and parameter names from trace.>",
        "evidences": [
            {{
                "type": "input_quality",
                "description": "Input structure quality examples with specific tool and parameter references",
                "examples": [
                    "<Example 1: '[ToolName] invocation at step [N]: Received complete input with fields <field1: value1, field2: value2>, properly formatted as [data_type]. Result: [successful_outcome]'>",
                    "<Example 2: '[ToolName] invocation at step [M]: Missing required field [field_name], received incomplete <partial_fields>. Result: [failure_or_retry]'>"
                ]
            }},
            {{
                "type": "tool_performance",
                "description": "Tools categorized by input quality with evidence from trace",
                "good_inputs": [
                    "<ToolName1: Step [N], all required parameters present ([list_parameters]), proper data types ([type1, type2]), enabled [specific_capability]>",
                    "<ToolName2: Step [M], well-formed [parameter_name] with [specific_format], resulted in [measurable_outcome]>"
                ],
                "poor_inputs": [
                    "<ToolName3: Step [X], missing [required_field], caused [specific_error/retry]>",
                    "<ToolName4: Step [Y], malformed [parameter_name] (expected [format], got [actual_format]), resulted in [failure_pattern]>"
                ]
            }},
            {{
                "type": "formatting_patterns",
                "description": "Consistent formatting approaches with specific examples from trace",
                "patterns": [
                    "<Pattern 1: [AgentName] consistently formatted [parameter_type] as [specific_structure] across [N invocations], enabling [specific_benefit]>",
                    "<Pattern 2: [Data_field] supplied in [format] for [ToolName1, ToolName2], reducing [specific_inefficiency]>"
                ]
            }}
        ],
        "performance_metrics": {{
            "total_inputs_analyzed": <int: total tool invocations in trace>,
            "well_structured_inputs": <int: count with all required fields and proper types>,
            "poorly_structured_inputs": <int: count with missing/malformed fields>,
            "structure_quality_rate": <float: well_structured / total>,
            "missing_fields_count": <int: total missing required parameters>,
            "type_mismatch_count": <int: parameters with wrong data types>
        }},
        "feedback": [
            {{
                "category": "input_improvement",
                "recommendation": "<State ONE specific input quality improvement using ACTUAL tool and parameter names from eval_context_info. Format: 'Ensure [SpecificTool] receives [required_field_name] in [expected_format] to prevent [observed_failure]. Currently missing in [X/Y invocations] at steps [list_steps].' Reference observable trace patterns.>",
                "impact": "<Quantify improvement: 'Would eliminate [N failures/retries]' or 'Expected to increase structure_quality_rate from [current%] to [target%]'>",
                "examples": "<Provide 2 concrete examples using ACTUAL tool/parameter names from eval_context_info. Format: 'Current input (step [N]): <ToolName>(incomplete_field1=value1) → [failure]. Improved input: <ToolName>(field1=value1, required_field2=value2, field3=value3) → [success]. Benefit: [Specific gain like 'eliminates retry in step N+1, saves 2 seconds']'. Reference schema from eval_context_info.>"
            }},
            {{
                "category": "format_optimization",
                "recommendation": "<State ONE specific formatting standardization using ACTUAL parameter names and data types. Format: 'Standardize [parameter_name] format across [ToolName1, ToolName2] to [specific_format] instead of mixed formats [format1, format2] observed in steps [X, Y, Z]. Apply pattern from [well_performing_tool].' Reference trace observations.>",
                "impact": "<Quantify improvement: 'Would reduce type_mismatch_count from [X] to [Y]' or 'Improve parameter consistency across [N tools]'>",
                "examples": "<Provide 2 concrete examples using ACTUAL tools and formats. Format: 'Current inconsistency: [ToolA] receives date as 'YYYY-MM-DD' (step 3), [ToolB] receives '2024-03-15T10:30:00' (step 7). Standardize to ISO8601 format across both. Observed issue: [ToolB] failed parsing at step 7. Expected benefit: [Eliminate format conversion overhead, prevent 1 parsing error per workflow]'. Reference eval_context_info schemas.>"
            }}
        ]
    }}
}}

**CRITICAL FEEDBACK REQUIREMENTS:**

**MANDATORY - Every feedback item must include:**
- At least 2 specific tool names from eval_context_info 
- At least 3 specific parameter/field names from tool schemas 
- Quantifiable metrics (e.g., "X missing fields", "Y type mismatches", "Z% quality rate")
- Observable trace evidence (cite step numbers and actual input values) [web:40]
- Expected data types and formats from eval_context_info schemas [web:42]
- Before/after input examples with concrete parameter structures [web:40]
- Impact statement with measurable outcome (failures eliminated, quality rate increase) [web:39]

**BANNED PHRASES (NEVER USE):**
- "improve input quality" / "better structure" / "optimize parameters"
- "ensure proper formatting" / "validate inputs" / "clean data"
- "appropriate fields" / "relevant parameters" / "suitable structure"
- "tools need better inputs" / "agents should format correctly"
- Any recommendation without specific tool/parameter names from eval_context_info
- Any example without actual field names, data types, or trace step references

**VALIDATION CHECKLIST - Each feedback recommendation must have:**
- [ ] References atleast 1 specific tool names from eval_context_info
- [ ] Includes atleast 1 actual parameter/field names from tool schemas
- [ ] Cites observable trace evidence (step numbers, actual values)
- [ ] Specifies expected data types/formats from schemas
- [ ] Quantifies current issue (X missing fields, Y type errors)
- [ ] Provides before/after input structure with actual parameters
- [ ] Includes impact with measurable improvement metric
- [ ] Zero generic/vague terminology

**EXAMPLE OF EXCELLENT FEEDBACK:**
{{
    "category": "input_improvement",
    "recommendation": "Ensure PolicyLookupTool receives required parameter 'member_id' (string, 10 digits) in all invocations. Currently missing in 4/7 calls (steps 3, 8, 12, 15), causing ParameterValidationError and forcing 2-step retry pattern",
    "impact": "Would increase well_structured_inputs from 3/7 (42.9%) to 7/7 (100%), eliminate 4 retry sequences, and reduce workflow execution time by 12 seconds",
    "examples": "Current input (step 3): PolicyLookupTool(policy_number='POL-12345') → ParameterValidationError: missing 'member_id'. Improved input: PolicyLookupTool(policy_number='POL-12345', member_id='1234567890', effective_date='2024-03-15') → Success: retrieved policy details in 0.8s. Schema reference (eval_context_info): PolicyLookupTool requires <member_id: string(10), policy_number: string, effective_date: date(ISO8601)>. Benefit: Eliminates retry at step 4, saves 3 seconds per occurrence."
}}

**EXAMPLE OF POOR FEEDBACK (DO NOT GENERATE):**
{{
    "category": "input_improvement",
    "recommendation": "Improve input quality by ensuring all required fields are present and properly formatted",
    "impact": "Would enhance overall system performance",
    "examples": "Make sure tools receive complete and correct parameters for better results"
}}

**ADDITIONAL REQUIREMENTS:**
1. **Schema Compliance**: Compare every input against tool schemas from eval_context_info 
2. **Type Validation**: Identify data type mismatches (expected vs actual) 
3. **Completeness Analysis**: Count missing required fields per tool invocation 
4. **Pattern Recognition**: Identify consistent formatting approaches that worked well 
5. **Cross-Tool Consistency**: Flag inconsistent parameter formats across similar tools
6. **Trace Evidence**: Cite specific step numbers and actual parameter values from trace_data

Sanitized Trace Data: {trace_data}

"""

AGENT_ROUTING_PROMPT_TEMPLATE = """
You are evaluating agent routing performance in a multi-agent workflow system.
        
AVAILABLE WORKFLOW RESOURCES:
Available Agents: {agents}
Available Tools: {tools}
Use ONLY these available resources for evaluation. Do not suggest tools or agents not present in this context.

SPECIAL INSTRUCTIONS FOR COMBINED TRACES:
- This trace represents an end-to-end workflow execution across multiple agents/steps
- Evaluate the complete flow from initial request to final output
- Consider inter-agent handoffs and data flow continuity
- Account for human-in-the-loop interactions if present
- Focus on the overall journey effectiveness, not individual trace segments

TRACE DATA TO ANALYZE:
{trace_data}

EVALUATION FOCUS:
Analyze the routing decisions, agent selection, and workflow coordination based on the actual agents and tools used in this specific trace.

EVALUATION CRITERIA:
1. Intent Detection Accuracy: How well the system identified the user's actual intent
2. Agent Selection Appropriateness: Whether the chosen agents were suitable for the identified tasks
3. Tool Utilization Effectiveness: How appropriately tools were selected and used by agents
4. Workflow Coordination: How well agents collaborated and passed information

EVIDENCE REQUIREMENTS:
- Extract specific agent names that were invoked
- Identify specific tools that were used by each agent
- Note the sequence of agent invocations
- Document any routing decisions or handoffs between agents
- Avoid mentioning trace IDs, trace combinations, or technical metadata
                                                            
FEEDBACK REQUIREMENTS:
- Identify how to improve the performance of agent routing evaluation
- Steps should be taken to increase the score

REASONING REQUIREMENTS:
- Provide workflow-specific analysis based on the actual agents and tools observed
- Explain why specific agent selections were appropriate or inappropriate for the task
- Discuss how tool usage contributed to or hindered task completion
- Address coordination patterns between different agents in the workflow

Expected JSON Output:
{{
    "score": <int 10-100>,
    "reasoning": "<Detailed workflow-specific assessment focusing on actual agent performance, tool usage effectiveness, and coordination quality observed in this trace>",
    "evidences": {{
        "agents_invoked": ["<agent_name_1>", "<agent_name_2>", "..."],
        "tools_used": ["<tool_name_1>", "<tool_name_2>", "..."],
        "routing_decisions": ["<specific routing decision 1>", "<specific routing decision 2>"],
        "coordination_patterns": ["<coordination pattern 1>", "<coordination pattern 2>"]
    }},
    "feedbacks": [
        "<Actionable feedback for improving agent selection based on observed patterns>",
        "<Specific recommendation for optimizing tool usage>",
        "<Suggestion for enhancing workflow coordination>",
        "<Example 1: provide examples similar to the context which could get higher scores>"
    ]
}}

**IMPORTANT: Base analysis on actual agent names, tool names, and workflow patterns from the trace. Avoid generic assessments.**

JSON:"""

HITL_PROMPT_TEMPLATE = """
You are evaluating how well the system honored human input, constraints, and oversight in a multi-agent workflow.
        
AVAILABLE WORKFLOW RESOURCES:
Available Agents: {agents}
Available Tools: {tools}
Use ONLY these available resources for evaluation. Do not suggest tools or agents not present in this context.

SPECIAL HANDLING FOR HITL NODE:
• This data contains Human-In-The-Loop (HITL) interactions: {hitl_info}
• Preserve all instructions, inputs, decisions provided by the human reviewer.
• Maintain chronological order between human actions and automated agent actions.
• Clearly distinguish human-initiated steps from agent-initiated steps.
• Ensure that all human guidance, overrides, corrections, and approvals remain intact.
• Capture the context under which the human stepped in (reason, trigger conditions, or workflow breakpoints).
• Reflect the impact of HITL input on subsequent agent decisions, state transitions, or outputs.
• Do NOT compress, summarize, or merge HITL steps—retain full fidelity of human involvement.

TRACE DATA TO ANALYZE:
{trace_data}

EVALUATION FOCUS:
Analyze human-in-the-loop interactions, constraint adherence, and oversight integration based on actual human inputs and system responses in this workflow.

EVALUATION CRITERIA:
1. Human Input Recognition: How well the system identified and processed human constraints/preferences
2. Constraint Adherence: Whether agents followed specified human constraints during execution
3. Oversight Integration: How effectively human oversight was incorporated at decision points
4. Feedback Incorporation: How well the system adapted based on human feedback

EVIDENCE REQUIREMENTS:
- Identify specific human constraints or preferences mentioned
- Document how different agents responded to human input
- Note any tools used specifically for human interaction or constraint checking
- Record instances where human oversight was requested or provided
- Avoid mentioning trace IDs, trace combinations, or technical metadata

FEEDBACK REQUIREMENTS:
- Identify how to improve the performance of HITL usage evaluation
- Steps should be taken to increase the score

REASONING REQUIREMENTS:
- Analyze specific instances where human constraints were honored or violated
- Evaluate how effectively different agents incorporated human guidance
- Assess the appropriateness of human oversight integration points
- Discuss the impact of human input on workflow outcomes

Expected JSON Output:
{{
    "score": <int 10-100>,
    "reasoning": "<Detailed analysis of how specific human inputs were handled, which agents demonstrated good constraint adherence, and how oversight was integrated into the workflow>",
    "evidences": {{
        "human_constraints": ["<constraint_1>", "<constraint_2>", "..."],
        "constraint_handling_agents": ["<agent_name_1>", "<agent_name_2>", "..."],
        "oversight_points": ["<oversight_instance_1>", "<oversight_instance_2>", "..."],
        "feedback_integration": ["<feedback_integration_1>", "<feedback_integration_2>", "..."]
    }},
    "feedbacks": [
        "<Specific recommendation for improving constraint recognition>",
        "<Actionable suggestion for better oversight integration>",
        "<Guidance for enhancing human feedback incorporation>",
        "<Example 1: provide examples similar to the context which could get higher scores>"
    ]
}}

**IMPORTANT: Focus on actual human interactions and constraint handling observed in this specific workflow.**

JSON:
"""

WORKFLOW_COMPLETION_PROMPT_TEMPLATE = """
You are evaluating workflow completion effectiveness in a multi-agent system.

AVAILABLE WORKFLOW RESOURCES:
Available Agents: {agents}
Available Tools: {tools}
Use ONLY these available resources for evaluation. Do not suggest tools or agents not present in this context.

SPECIAL INSTRUCTIONS FOR COMBINED TRACES:
- This trace represents an end-to-end workflow execution across multiple agents/steps
- Evaluate the complete flow from initial request to final output
- Consider inter-agent handoffs and data flow continuity
- Account for human-in-the-loop interactions if present
- Focus on the overall journey effectiveness, not individual trace segments

TRACE DATA TO ANALYZE:
{trace_data}

EVALUATION FOCUS:
Analyze how effectively the agent workflow completed the intended task, focusing on agent coordination, tool effectiveness, and outcome quality.

EVALUATION CRITERIA:
1. Task Objective Achievement: How well the workflow accomplished the stated goal
2. Agent Coordination Effectiveness: Quality of collaboration between different agents
3. Tool Selection and Usage: Appropriateness and effectiveness of tool choices
4. Outcome Quality: Whether the final result meets expectations and requirements

EVIDENCE REQUIREMENTS:
- Document the original task/objective from the trace
- Identify which agents contributed to task completion
- List specific tools used and their contribution to the outcome
- Note the sequence and coordination of agent activities
- Record the final outcome or deliverable
- Avoid mentioning trace IDs, trace combinations, or technical metadata

REASONING REQUIREMENTS:
- Compare the intended task with the actual outcome achieved
- Evaluate the effectiveness of the agent sequence and coordination
- Assess whether tool usage was optimal for the task requirements
- Analyze the quality and completeness of the final deliverable

FEEDBACK REQUIREMENTS:
- Identify how to improve the performance of workflow completion evaluation 
- Steps should be taken to increase the score

**CRITICAL FEEDBACK REQUIREMENTS:**

**MANDATORY - Every feedback item must include:**
- At least 2 specific agent names from the trace that were involved
- At least 2 specific tool names used in the workflow execution
- At least 3 specific parameter/field names or workflow attributes referenced
- Quantifiable metrics (e.g., "X coordination gaps", "Y tool invocations", "Z% completion rate")
- Observable trace evidence (cite specific agent actions, tool calls, and actual values)
- Expected workflow patterns and actual execution patterns comparison
- Before/after workflow examples with concrete agent/tool interactions
- Impact statement with measurable outcome (e.g., "would reduce completion time by X", "eliminate Y handoff failures")

**BANNED PHRASES (NEVER USE):**
- "improve workflow quality" / "better coordination" / "optimize execution"
- "ensure proper task completion" / "validate workflows" / "clean execution"
- "appropriate agents" / "relevant tools" / "suitable coordination"
- "agents need better coordination" / "workflow should be improved"
- Any recommendation without specific agent/tool names from the trace
- Any example without actual workflow steps, agent names, or tool invocations

**VALIDATION CHECKLIST - Each feedback recommendation must have:**
- [ ] References at least 2 specific agent names from the trace
- [ ] Includes at least 2 actual tool names used in workflow
- [ ] Includes at least 3 workflow attributes (parameters, states, outputs)
- [ ] Cites observable trace evidence (agent actions, tool calls, specific values)
- [ ] Specifies expected workflow patterns vs actual execution
- [ ] Quantifies current issue (X coordination gaps, Y failed handoffs, Z retries)
- [ ] Provides before/after workflow structure with actual agent/tool interactions
- [ ] Includes impact with measurable improvement metric
- [ ] Zero generic/vague terminology

**EXAMPLE OF EXCELLENT FEEDBACK:**
"Ensure DataRetrievalAgent passes 'customer_id' (string, UUID format) to AnalysisAgent in handoff at step 5. Currently missing in 3/5 workflow executions, causing AnalysisAgent to invoke CustomerLookupTool unnecessarily, adding 4.2s latency per occurrence. DataRetrievalAgent should extract customer_id from InitialQueryTool response (field: 'primary_customer_uuid') and include in context payload. Impact: Would eliminate 3 redundant CustomerLookupTool invocations per workflow, reduce average completion time from 18.7s to 14.5s (22% improvement). Example: Current execution (step 5): DataRetrievalAgent → AnalysisAgent with context={{'query_result': [...]}}, missing 'customer_id' → AnalysisAgent invokes CustomerLookupTool at step 6. Improved: DataRetrievalAgent → AnalysisAgent with context={{'query_result': [...], 'customer_id': 'a7f3c2e1-9b4d'}}, no lookup needed."

**EXAMPLE OF POOR FEEDBACK (DO NOT GENERATE):**
"Improve agent coordination by ensuring proper data handoffs and complete context sharing for better results"

**ADDITIONAL FEEDBACK REQUIREMENTS:**
1. **Coordination Analysis**: Identify specific handoff points between agents and analyze data completeness
2. **Tool Usage Patterns**: Compare tool invocation sequences against optimal workflow patterns
3. **Outcome Verification**: Quantify gap between expected vs actual workflow deliverables
4. **Timing Analysis**: Identify workflow bottlenecks with specific agent/tool timing measurements
5. **Cross-Agent Consistency**: Flag inconsistent data formats/parameters across agent boundaries
6. **Trace Evidence**: Cite specific step numbers, agent names, tool calls, and actual parameter values

Expected JSON Output:
{{
    "score": <int 10-100>,
    "reasoning": "<Comprehensive analysis of task completion effectiveness, focusing on how specific agents and tools contributed to achieving the objective, and the quality of the final outcome>",
    "evidences": {{
        "task_objective": "<original task from trace>",
        "contributing_agents": ["<agent_name_1>", "<agent_name_2>", "..."],
        "tools_utilized": ["<tool_name_1>", "<tool_name_2>", "..."],
        "coordination_sequence": ["<step_1>", "<step_2>", "..."],
        "final_outcome": "<description of actual deliverable/result>"
    }},
    "feedbacks": [
        "<Feedback 1: Must include at least 2 agent names, 2 tool names, 3 workflow attributes, quantified metrics, trace evidence with step numbers, before/after examples, and measurable impact>",
        "<Feedback 2: Must follow same requirements with different workflow aspect>",
        "<Feedback 3: Must cite specific coordination patterns from trace>",
        "<Feedback 4: Must include timing/performance measurements from actual execution>"
    ]
}}

**IMPORTANT: Focus on actual task completion effectiveness using specific agent names, tool names, and workflow outcomes from this trace. All feedback must be actionable, measurable, and backed by concrete trace evidence.**

JSON:"""

CUSTOM_METRIC_PROMPT_TEMPLATE = """
You are evaluating workflow performance based on user-defined custom metrics and evaluation criteria.

AVAILABLE WORKFLOW RESOURCES:
Available Agents: {agents}
Available Tools: {tools}
Use ONLY these available resources for evaluation. Do not suggest tools or agents not present in this context.

SPECIAL INSTRUCTIONS FOR COMBINED TRACES:
- This trace represents an end-to-end workflow execution across multiple agents/steps
- Evaluate the complete flow from initial request to final output
- Consider inter-agent handoffs and data flow continuity
- Account for human-in-the-loop interactions if present
- Focus on the overall journey effectiveness, not individual trace segments

TRACE DATA TO ANALYZE:
{trace_data}

CUSTOM EVALUATION CRITERIA:
{custom_instructions}

EVALUATION FOCUS:
Evaluate the workflow execution specifically against the user-defined custom metrics provided above. 
Focus on measuring adherence to the custom instructions and requirements.

EVALUATION APPROACH:
1. Parse and understand each custom metric/criterion provided
2. Identify evidence in the trace that relates to each criterion
3. Assess compliance level for each criterion
4. Provide an overall score based on how well the workflow met the custom requirements

EVIDENCE REQUIREMENTS:
- Map specific trace events/actions to each custom criterion
- Document instances of compliance and non-compliance
- Extract relevant agent actions and tool usage that relate to custom metrics
- Avoid mentioning trace IDs, trace combinations, or technical metadata

REASONING REQUIREMENTS:
- Explain how well each custom criterion was met
- Provide specific examples from the trace
- Justify the overall score based on criterion-by-criterion analysis
- Highlight areas where custom requirements were exceeded or missed

FEEDBACK REQUIREMENTS:
- Identify how to improve the performance of custom metric evaluation 
- Steps should be taken to increase the score

Expected JSON Output:
{{
    "score": <int 10-100>,
    "reasoning": "<Detailed analysis of how well the workflow met the user-defined custom evaluation criteria, with specific evidence from the trace>",
    "evidences": {{
        "evaluated_criteria": ["<criterion_1>", "<criterion_2>", "..."],
        "compliance_instances": ["<compliance_evidence_1>", "<compliance_evidence_2>", "..."],
        "non_compliance_instances": ["<non_compliance_1>", "<non_compliance_2>", "..."],
        "relevant_agents": ["<agent_name_1>", "<agent_name_2>", "..."],
        "relevant_tools": ["<tool_name_1>", "<tool_name_2>", "..."]
    }},
    "criterion_breakdown": [
        {{
            "criterion": "<criterion_description>",
            "met": <true/false>,
            "evidence": "<specific evidence>",
            "score_impact": "<how this affected the overall score>"
        }}
    ],
    "feedbacks": [
        "<Specific recommendation for better meeting custom criterion 1>",
        "<Actionable suggestion for improving compliance with criterion 2>",
        "<Guidance for exceeding custom metric requirements>",
        "<Example 1: provide examples similar to the context which could get higher scores>"
    ]
}}

**IMPORTANT: Focus exclusively on the user-defined custom criteria. Base all assessments on actual evidence from the trace.**

JSON:"""

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