## Updates

1. **Controller Argument Added**  
   I added a `controller` argument in `run_webqsp_llm.py`.  
   - **Iterative mode**: First retrieves the relation from the LLM, then fetches the corresponding entity.  
   - **Direct mode**: Sends both relation and entity together to the LLM at each step.  
   - Iterative mode is mostly aligned with Eric’s original approach.

2. **Changes in Iterative Mode**  
   - Updates are mainly in the controller and prompt structure.  
   - The pipeline now:
     - First identifies the initial entity  
     - Uses separate **system** and **user prompts**  
   - Each step is divided into two phases:
     - **Relation fetch phase** (separate system + user prompts)  
     - **Entity fetch phase** (separate system + user prompts)  
   - Previously, a single system/user prompt pair was shared across all stages; this is now split for better clarity and control.

3. **Entity Representation Update**  
   - During entity fetch, we now send **human-readable entity names** when available instead of IDs.  
   - Issue: Most entities in the current dataset lack human-readable names.  
   - Current behavior:
     - Use human-readable name if available  
     - Otherwise fall back to KG ID  
   - The input list may contain a mix of readable names and KG IDs.

4. **Separate Controller for Direct Mode**  
   - A separate controller file is used for the direct approach.  
   - This ensures the iterative implementation remains unchanged.

5. **Model Performance Variability**  
   - Performance varies significantly across models:
     - Gemini 3.1-flash: ~6/10 correct (iterative)  
     - Gemini 2.5-flash-lite: ~3/10 correct (iterative)  
   - Results are **non-deterministic**; repeated runs yield different outcomes.  
   - Direct mode currently achieves ~4/10 (first 10 samples).  
   - Although lower, it is more suitable for caching and can likely be improved with further tuning.

6. **Frontier Issue in Intermediate Steps**  
   - Many runs fail to find a frontier during intermediate steps.  
   - Likely caused by limitations in the current KG structure.

7. **Caching Implementation**  
   - Implemented **LRU**, **LFU**, and **Oracle** caching policies.  
   - Not yet tested — requires trace collection before evaluation.  
   - Additional parameters added in `run_webqsp_llm.py`:
     - If `cache_size == 0`, defaults to uncached backend  
   - Currently, caching is only implemented for the direct controller and has not been tested yet.

8. **Code Status**  
   - Changes have not been merged into the main branch yet.  
   - Please review and let me know if any modifications are needed.