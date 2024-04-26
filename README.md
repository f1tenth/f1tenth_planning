# f1tenth_planning
Central repo for all motion planners created for F1TENTH.

# Basic API

## Controllers
### Init

init should take a configuration of some sort and not env or task (might have unecessary leakage to controller in benchmark).
Up to users to make sure configuration matches the env and the task.

\_\_init\_\_(self, config)

### Plan

1. plan(env_obs) -> (speed, steer)
2. plan(env_obs) -> (accl, steer_vel)

TODOs:
- Needs to know what control output type to use.
- Needs to know what env obs type controller is getting, potentially throw error if no key sensor type is found.

### Update config

should take the same input as init

update(self, config)

## Planners
### init
See controller init notes.

\_\_init\_\_(self, config)

### Plan

1. plan(env_obs) -> (speed, steer)
2. plan(env_obs) -> (accl, steer_vel)
3. plan(env_obs) -> path
4. plan(env_obs) -> trajectory (path + vel)

TODOs:
- Needs to know what output type to use.
- Needs to know what env obs type controller is getting, potentially throw error if no key sensor type is found.

### Update config

see controller update notes.

update(self, config)


## Trajectory dataclass
### Fields
- type: str
- positions: [(x,y)]
- headings: [Optional(theta)]
- velocities: [Optional(v)]
- accelerations: [Optional(a)]
- steering_angles: [Optional(delta)]
- steering_velocities: [Optional(delta_dot)]

### Initialization
- from file (.csv)
- from planner or controllers (fill fields in)

### Types
- positions only
- poses only
- positions + velocities
- poses + velocities
- steering_angles + velocities
- steering_velocities + accelerations