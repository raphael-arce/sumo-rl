import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
net = sumolib.net.readNet('nets/charlottenburg/actuated.net.xml')
import numpy as np
from gym import spaces


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, use_pressure=False):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        # self.yellow_time = yellow_time # patch: we use the yellow time from the net definition further below
        self.yellow_time = 0
        self.red_time = 0
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.has_red = False
        self.is_red = False
        self.is_green = False
        self.time_since_last_phase_change = 0
        self.use_pressure = use_pressure
        self.next_action_time = 0
        self.last_measure = 0.0
        self.last_reward = None
        self.phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0].phases
        self.different_phases = {
            'g': 0,
            'y': 0,
            'r': 0
        }
        self.num_green_phases = 0
        self.green_phase_mapper = {}
        for index, phase in enumerate(self.phases):
            s = phase.state
            if 'G' in s or 'g' in s:
                self.num_green_phases += 1
                self.different_phases['g'] += 1
                self.green_phase_mapper[self.num_green_phases-1] = index
            elif 'r' in s and 'G' not in s and 'g' not in s and 'y' not in s:
                self.different_phases['r'] += 1
                if phase.duration > 0:
                    if self.red_time != 0:
                        if self.red_time != phase.duration:
                            raise Exception(f'{self.id} has different red times!')
                    else:
                        self.has_red = True
                        self.red_time = phase.duration
            elif 'y' in s:
                self.different_phases['y'] += 1
                if phase.duration > 0:
                    if self.yellow_time != 0:
                        if self.yellow_time != phase.duration:
                            raise Exception(f'{self.id} has different yellow times!')
                    else:
                        self.yellow_time = phase.duration
            else:
                raise Exception(f'{self.id} has another state {s} within phases!')

        self.num_different_phases = 0
        for diff_phase in self.different_phases:
            if self.different_phases[diff_phase] > 0:
                self.num_different_phases += 1

        #self.num_green_phases = len(self.phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order


        if len(self.phases) > 3: # ignore traffic lights without intersection (e.g. just a light for pedestrians to cross)
            copy = []
            alreadyVisited = []
            offset = 0
            for i, lane in enumerate(self.lanes):
                self.verify_and_append_incoming_lanes(copy, lane, i+offset, alreadyVisited)
            if set(self.lanes) != set(copy):
                print(f'intersection {self.id} had at least one incoming lane smaller than 23 meters, extended it with:')
                print(set(copy) - set(self.lanes))
                self.lanes = copy


        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.neighbors = {}

        """
        Default observation space is a vector R^(#greenPhases + 2 * #lanes)
        s = [current phase one-hot encoded, observation metric]
        You can change this by modifing self.observation_space and the method _compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds
        """
        self.observation_space = spaces.Box(
            low=np.zeros(self.num_green_phases + 2*len(self.lanes)),
            high=np.ones(self.num_green_phases + 2*len(self.lanes)))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                       # Green Phase
            spaces.Discrete(10)                                           # Metric
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

        logic = traci.trafficlight.Logic("new-program"+self.id, 0, 0, phases=self.phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    def verify_and_append_incoming_lanes(self, lanes, lane, offset, alreadyVisited):
        if lane not in lanes and lane not in alreadyVisited:
            alreadyVisited.append(lane)
            lanes.insert(offset, lane)
            length = traci.lane.getLength(lane)
            if length < 23:
                incomingLanes = net.getLane(lane).getIncoming()
                for incomingLane in incomingLanes:
                    offset += 1
                    self.verify_and_append_incoming_lanes(lanes, incomingLane.getID(), offset, alreadyVisited)

    def set_neighbors(self, traffic_signals):
        for ts in traffic_signals:
            lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(ts)))
            sharedLanes = 0
            for lane in lanes:
                if lane in self.out_lanes:
                    sharedLanes += 1
            if sharedLanes > 0:
                self.neighbors[ts] = traffic_signals[ts]

        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),
            *(spaces.Discrete(10) for _ in range(1 + len(self.neighbors)))  # own metric + metric of neighbor TS
        ))


    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step
    
    def update(self):
        self.time_since_last_phase_change += 1

        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            traci.trafficlight.setPhase(self.id, (self.phase + 1) % len(self.phases))
            self.is_yellow = False
            self.time_since_last_phase_change = 0

            s = traci.trafficlight.getRedYellowGreenState(self.id)
            if 'G' in s or 'g' in s:
                self.is_green = True
                self.is_red = False
                self.is_yellow = False
            elif 'r' in s and 'G' not in s and 'g' not in s and 'y' not in s:
                self.is_green = False
                self.is_red = True
                self.is_yellow = False
            elif 'y' in s:
                self.is_green = False
                self.is_red = False
                self.is_yellow = True

        if self.is_red and self.time_since_last_phase_change == self.red_time:
            traci.trafficlight.setPhase(self.id, (self.phase + 1) % len(self.phases))
            self.is_red = False
            self.time_since_last_phase_change = 0

            s = traci.trafficlight.getRedYellowGreenState(self.id)
            if 'G' in s or 'g' in s:
                self.is_green = True
                self.is_red = False
                self.is_yellow = False
            elif 'r' in s and 'G' not in s and 'g' not in s and 'y' not in s:
                self.is_green = False
                self.is_red = True
                self.is_yellow = False
            elif 'y' in s:
                self.is_green = False
                self.is_red = False
                self.is_yellow = True

    def set_next_phase(self, new_phase):
        """
        Sets the next phase if the given new_phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases] 
        """
        if self.phase == self.green_phase_mapper[new_phase] or self.time_since_last_phase_change < self.min_green + self.yellow_time + self.red_time:
            traci.trafficlight.setPhase(self.id, self.phase)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            traci.trafficlight.setPhase(self.id, (self.phase + 1) % len(self.phases))
            self.next_action_time = self.env.sim_step + self.delta_time
            self.time_since_last_phase_change = 0

            s = traci.trafficlight.getRedYellowGreenState(self.id)
            if 'G' in s or 'g' in s:
                self.is_green = True
                self.is_red = False
                self.is_yellow = False
            elif 'r' in s and 'G' not in s and 'g' not in s and 'y' not in s:
                self.is_green = False
                self.is_red = True
                self.is_yellow = False
                self.next_action_time += self.red_time
            elif 'y' in s:
                self.is_green = False
                self.is_red = False
                self.is_yellow = True
                self.next_action_time += self.yellow_time
    
    def compute_observation(self):
        phase_id = [1 if self.phase//2 == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        if self.use_pressure:
            print('pressure')
            pressure = self.get_pressure()
            for neighbor in self.neighbors:
                print('neighbors pressure')
                pressure += self.neighbors[neighbor].get_pressure()
            return np.array(phase_id + pressure)
        else:
            print('Q+D')
            density = self.get_lanes_density()
            queue = self.get_lanes_queue()
            for neighbor in self.neighbors:
                print('neighbors Q+D')
                density += self.neighbors[neighbor].get_lanes_density()
                queue += self.neighbors[neighbor].get_lanes_queue()
            return np.array(phase_id + density + queue)

            
    def compute_reward(self):
        self.last_reward = self._waiting_time_reward()
        return self.last_reward
    
    def _pressure_reward(self):
        return -self.get_pressure()[0]

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return - (sum(self.get_stopped_vehicles_num()))**2

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_pressure(self):
        sum_in = 0
        for lane in self.lanes:
            sum_in += traci.lane.getLastStepVehicleNumber(lane)
        sum_out = 0
        for lane in self.out_lanes:
            sum_out += traci.lane.getLastStepVehicleNumber(lane)
        return [abs(sum_in - sum_out)]

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]
    
    def get_total_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list
