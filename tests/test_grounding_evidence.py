"""Geometry and parsing shared by unified element inference."""
from grounding.evidence import containing_candidates, proposal_info
from grounding.coordinates import parse_coordinate_tail

def observations():
    row=dict(index=0,action_key='task_action',task_id='task',page_size=[100,100],
        candidate_bboxes=[[0,0,100,100],[20,20,10,10],[20,20,10,10]],
        windows=[dict(top=0,centres={'0':[500,500],'1':[250,250],'2':[250,250]})])
    points=dict(action_key=row['action_key'],windows=[
        dict(top=0,crop_size=[100,100],coordinate=[250,250],log_probability=-1.,token_ids=[1,2,3])])
    return row,points


def test_bbox_boundary_and_overlap_keep_public_rank():
    row,_=observations()
    assert containing_candidates(row,[20,20])==[0,1,2]
    assert containing_candidates(row,[30,30])==[0]
    assert containing_candidates(row,[100,100])==[]
    _,existing,new=proposal_info(row,observations()[1])
    assert existing=={1} and new==[0,2]


def test_coordinate_parser_stops_at_bracket_and_rejects_out_of_range():
    assert parse_coordinate_tail('205, 280]}}</tool_call>')==[205,280]
    assert parse_coordinate_tail('1000, 2]') is None
    assert parse_coordinate_tail('-1, 2]') is None
