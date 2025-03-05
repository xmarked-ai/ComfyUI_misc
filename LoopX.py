import json
import os
import gc
import time
import re
from datetime import datetime
from random import randrange as rnd

#ComfyUI Related Imports
import torch
import nodes
import folder_paths
import comfy.utils
import comfy.model_management as mem_manager
from server import PromptServer

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
    from comfy_execution.graph import ExecutionBlocker
except:
    ExecutionBlocker = None
    GraphBuilder = None

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

MAX_SLOTS = 2
VYKOSX_STORAGE_DATA = {}

class LoopOpenX:
    max_slots = MAX_SLOTS

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "start": ("INT", {"default": 1,"defaultInput": False,"tooltip": "The initial value of the loop counter"}),
                "step": ("INT", {"default": 1,"defaultInput": False,"tooltip": "How much the loop counter gets increased or decreased by on each iteration"}),
                "end": ("INT", {"default": 10,"defaultInput": False, "tooltip": "The value at which the loop should stop"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "original_id": "INT",
                "index_override": "INT",  # Для передачи индекса между итерациями
            }
        }

        return inputs

    FUNCTION = "loop"
    CATEGORY = "xmtools/nodes"

    RETURN_TYPES = tuple([any_type, "INT"])
    RETURN_NAMES = tuple(["LOOP", "index"])

    def loop(self, start, step, end, **kwargs):
        original_id = kwargs.get("original_id", None)

        if original_id is None:
            original_id = kwargs['unique_id']

        index = start if original_id is None else kwargs.get("index_override", start)

        finished = (( end - index ) <= 0) if step >= 0 else (( end - index ) >= 0)
        loop_status = {"id":original_id, "start":start, "end":end, "step":step, "index":index, "finished":finished, "last_id":kwargs['unique_id']}

        Vars = loop_status
        for key, value in VYKOSX_STORAGE_DATA.items(): Vars[key] = value

        if not finished:
            return (loop_status, index)
        else:
            if ExecutionBlocker is None:
                raise Exception("Unable to block execution, please update your ComfyUI to enable this functionality!")
            else:
                return (ExecutionBlocker(None), ExecutionBlocker(None))

class LoopCloseX:
    def __init__(self):
        pass

    max_slots = MAX_SLOTS+1

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "LOOP": (any_type, {"forceInput": True,"tooltip": "Connect to the LOOP output of a [Loop Open] node to establish a Loop"}),
            },
            "optional": {
                "data": (any_type, {"forceInput": True, "tooltip": "Connect nodes here to execute them within the loop"}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

        return inputs

    CATEGORY = "xmtools/nodes"
    FUNCTION = "loop"
    RETURN_TYPES = tuple(["BOOLEAN", "INT"])
    RETURN_NAMES = tuple(["FINISHED?", "index"])

    def find_subnodes(self, node_id, dynprompt, node_data):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for key, value in node_info["inputs"].items():
            if is_link(value):
                parent_id = value[0]
                if parent_id not in node_data:
                    node_data[parent_id] = []
                    self.find_subnodes(parent_id, dynprompt, node_data)
                node_data[parent_id].append(node_id)

    def get_subnodes(self, node_id, node_data, subnodes):
        if node_id not in node_data:
            return
        for child_id in node_data[node_id]:
            if child_id not in subnodes:
                subnodes[child_id] = True
                self.get_subnodes(child_id, node_data, subnodes)

    def loop(self, LOOP, dynprompt=None, unique_id=None, **kwargs):
        if LOOP['finished']:
            return (True, LOOP['index'])
        else:
            this_node = dynprompt.get_node(unique_id)

            self.find_subnodes(unique_id, dynprompt, node_data:={}) # Get list of all nodes connected to the loop
            self.get_subnodes(open_node:=LOOP['last_id'], node_data, subnodes:={}) #Find only the nodes that are within both Open and Close Loop

            subnodes[unique_id] = True; subnodes[open_node] = True

            graph = GraphBuilder()

            if graph is None:
                raise Exception("Unable to create Loop system, this ComfyUI version is too old to support it.\nPlease update your ComfyUI to be able to utilize Loops!")

            for node_id in subnodes:
                original_node = dynprompt.get_node(node_id)
                node = graph.node(original_node["class_type"], "R" if node_id == unique_id else node_id)
                node.set_override_display_id(node_id)

            for node_id in subnodes: #Iterate over each of our subnodes
                original_node = dynprompt.get_node(node_id)
                node = graph.lookup_node("R" if node_id == unique_id else node_id)

                for key, value in original_node["inputs"].items(): #Recreate the inputs for all the subnodes
                    if is_link(value) and value[0] in subnodes:
                        parent = graph.lookup_node(value[0])
                        node.set_input(key, parent.out(value[1]))
                    else:
                        node.set_input(key, value)

            #Recreate the inputs of our new Loop Open node
            new_open = graph.lookup_node(open_node)
            new_open.set_input("index_override", LOOP['index']+LOOP['step'])
            new_open.set_input("original_id", LOOP['id'])

            new_subnode_graph = graph.lookup_node("R")
            result = [new_subnode_graph.out(0), new_subnode_graph.out(1)] # LOOP и index

            return { "result": tuple(result), "expand": graph.finalize(), }

class IfConditionX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "A": (any_type, {"forceInput": True, "lazy": True, "tooltip": "Condition to evaluate (True/False)"}),
                "TRUE_IN": (any_type, {"forceInput": True, "lazy": True, "tooltip": "Input to forward if condition is True"}),
                "FALSE_IN": (any_type, {"forceInput": True, "lazy": True, "tooltip": "Input to forward if condition is False"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("result",)
    FUNCTION = "evaluate"
    CATEGORY = "xmtools/nodes"

    def check_lazy_status(self, A=None, TRUE_IN=None, FALSE_IN=None, unique_id=0):
        if A is None:
            return ["A"]
        else:
            condition = bool(A) if isinstance(A, (int, float)) else A
            if condition:
                return ["TRUE_IN"]
            else:
                return ["FALSE_IN"]

    def evaluate(self, A=None, TRUE_IN=None, FALSE_IN=None, unique_id=0):
        if isinstance(A, (int, float)):
            condition = bool(A)
        else:
            condition = A

        if condition:
            return (TRUE_IN,)
        else:
            return (FALSE_IN,)


LOOPX_CLASS_MAPPINGS = {
    "LoopOpenX": LoopOpenX,
    "LoopCloseX": LoopCloseX,
    "IfConditionX": IfConditionX,
}

LOOPX_NAME_MAPPINGS = {
    "LoopOpenX": "Loop Open X",
    "LoopCloseX": "Loop Close X",
    "IfConditionX": "If Condition X",
}
