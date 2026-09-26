def start_active_events(active_states: dict, current_ids: set, t_sec: float):
    """Starts tracking a timestamp for any new ID that meets an event condition."""
    for tid in current_ids:
        if tid not in active_states:
            active_states[tid] = t_sec

def close_finished_events(active_states: dict, current_ids: set, t_sec: float, event_name: str, min_duration: float = 0.5) -> list[list]:
    """Closes events for IDs that no longer meet the condition and formats the output."""
    completed_events = []
    stopped_ids = [tid for tid in active_states if tid not in current_ids]
    
    for tid in stopped_ids:
        start_sec = active_states.pop(tid)
        duration = t_sec - start_sec
        if duration >= min_duration:
            completed_events.append([round(start_sec, 2), round(t_sec, 2), event_name])
            
    return completed_events