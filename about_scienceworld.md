## 启动指令
```shell
python examples/human.py --task-num=3 --num-episodes=5
```

## 游戏示例
Gold Path:['open door to bedroom', 'go to bedroom', 'open door to hallway', 'go to hallway', 'open door to workshop', 'go to workshop', 'open door to hallway', 'go to hallway', 'open door to living room', 'go to living room', 'open door to hallway', 'go to hallway', 'open door to art studio', 'go to art studio', 'look around', 'pour cup containing blue paint in art studio in jug', 'pour cup containing yellow paint in art studio in jug', 'mix jug', 'look around', 'focus on green paint', 'wait1']

Task Name: chemistry-mix-paint-secondary-color

Variation: 0 / 36

Task Description: Your task is to use chemistry to create green paint. When you are done, focus on the green paint.

This room is called the hallway. In it, you see: 
- the agent
- a substance called air
- a picture
You also see:
- A door to the art studio (that is closed)
- A door to the bedroom (that is closed)
- A door to the greenhouse (that is closed)
- A door to the kitchen (that is closed)
- A door to the living room (that is closed)
- A door to the workshop (that is closed)

Reward: 0

Score: 0

isCompleted: False

'help' lists valid action templates, 'objects' lists valid objects, use <tab> to list valid actions. 

'goals' lists progress on subgoals.

type 'exit' to quit.

> 

## 可行动作集合
### Possible actions: 
- activate OBJ
- close OBJ
- connect OBJ to OBJ
- deactivate OBJ
- disconnect OBJ
- dunk OBJ in OBJ
- eat OBJ
- flush OBJ
- focus on OBJ
- go OBJ
- inventory
- look around
- look at OBJ
- look in OBJ
- mix OBJ
- move OBJ to OBJ
- open OBJ
- pick up OBJ
- pour OBJ in OBJ
- put down OBJ
- read OBJ
- reset task
- task
- use OBJ on OBJ
- wait
- wait1

### Possible objects (one referent listed per object): 
- agent
- air
- art studio
- art studio door
- bedroom
- bedroom door
- door to greenhouse
- door to kitchen
- door to living room
- door to workshop
- greenhouse
- hallway
- kitchen
- living room
- picture
