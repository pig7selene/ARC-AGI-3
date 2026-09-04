from __future__ import annotations
from typing import Any
def build_context(observation, abstraction, diff, memory, valid_actions) -> dict[str,Any]:
    return {"observation":{"width":observation.width,"height":observation.height,"level":observation.level,"state_hash":observation.state_hash,
      "color_frequencies":abstraction.color_frequencies,"symmetry":abstraction.symmetry,"grid":observation.grid},
      "objects":[{"id":o.id,"color":o.color,"bbox":o.bbox,"area":o.area,"centroid":o.centroid,"border":o.border_contact,"shape":o.shape_signature[:32]} for o in abstraction.objects],
      "last_transition":diff.summary() if diff else None,"memory":memory.compact(),"valid_actions":[getattr(a,"value",str(a)) for a in valid_actions],"no_op_actions":sorted(memory.no_op_actions),
      "current_plan":[p.as_dict() if hasattr(p,"as_dict") else str(p) for p in memory.current_plan]}
