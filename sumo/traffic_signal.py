import os
import sys
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, phases):
        self.id = ts_id
        self.env = env
        self.time_on_phase = 0.0
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.num_green_phases = len(phases) // 2
        self.lanes = list(
            dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # remove duplicates and keep order
        self.edges = self._compute_edges()
        self.edges_capacity = self._compute_edges_capacity()

        self.veh_list_pre = []
        self.wt_car = dict()
        self.wt_firetruck = dict()
        self.veh_travel_record = dict()

        self.neighbors = []
        for i in range(len(self.edges)):
            from_node = self.edges[i][0].split('-')[1]
            if len(from_node) > 3:
                self.neighbors.append(None)
            else:
                self.neighbors.append(from_node)

        logic = traci.trafficlight.Logic("new-program", 0, 0, phases=phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    def get_neighbor(self):
        return self.neighbors

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current
        :param new_phase: (int) Number between [0..num_green_phases]
        """
        new_phase *= 2
        if self.phase == new_phase or self.time_on_phase < self.min_green:
            self.time_on_phase += self.delta_time
            self.green_phase = self.phase
        else:
            self.time_on_phase = self.delta_time - self.yellow_time
            self.green_phase = new_phase
            traci.trafficlight.setPhase(self.id, self.phase + 1)  # turns yellow

    def update_phase(self):
        """
        Change the next green_phase after it is set by set_next_phase method
        """
        traci.trafficlight.setPhase(self.id, self.green_phase)

    def _compute_edges(self):
        """
        return: Dict green phase to edge id
        """
        return {p: self.lanes[p * 3:p * 3 + 3] for p in range(self.num_green_phases)}  # two lanes per edge

    def _compute_edges_capacity(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return {
            p: sum([traci.lane.getLength(lane) for lane in self.edges[p]]) / vehicle_size_min_gap for p in
        range(self.num_green_phases)
        }

    def get_density(self):
        return [sum([traci.lane.getLastStepVehicleNumber(lane) for lane in self.edges[p]]) / self.edges_capacity[p] for
                p in range(self.num_green_phases)]

    def get_stopped_density(self):
        return [sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[p]]) / self.edges_capacity[p] for
                p in range(self.num_green_phases)]

    def get_stopped_vehicles_num(self):
        return [sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.edges[p]]) for p in
                range(self.num_green_phases)]

    def get_avg_wait_time_priority_all(self):
        for p in range(self.num_green_phases):
            veh_list = self.get_veh_list(p)
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                # acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                # #计算的是车辆在某一个车道上的等待时间
                # if veh not in self.env.vehicles:
                #     self.env.vehicles[veh] = {veh_lane: acc}
                # else:
                #     self.env.vehicles[veh][veh_lane] = acc - sum(
                #         [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                if traci.vehicle.getTypeID(veh) == 'FireTruck':
                    self.wt_firetruck[veh] = self.env.vehicles[veh][veh_lane]
                else:
                    self.wt_car[veh] = self.env.vehicles[veh][veh_lane]
        avg_wt_car = 0.0 if len(self.wt_car)<1 else np.mean([self.wt_car[c] for c in self.wt_car])
        avg_wt_firetruck = 0.0 if len(self.wt_firetruck)<1 else np.mean([self.wt_firetruck[f] for f in self.wt_firetruck])

        return avg_wt_car, avg_wt_firetruck

    def get_avg_wait_time_priority(self):
        wait_time_car_list = []
        wait_time_firetruck_list = []
        for p in range(self.num_green_phases):
            veh_list = self.get_veh_list(p)
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                # acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                # #计算的是车辆在某一个车道上的等待时间
                # if veh not in self.env.vehicles:
                #     self.env.vehicles[veh] = {veh_lane: acc}
                # else:
                #     self.env.vehicles[veh][veh_lane] = acc - sum(
                #         [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                if traci.vehicle.getTypeID(veh) == 'FireTruck':
                    wait_time_firetruck_list.append(self.env.vehicles[veh][veh_lane])
                    self.wt_firetruck[veh] = self.env.vehicles[veh][veh_lane]
                else:
                    wait_time_car_list.append(self.env.vehicles[veh][veh_lane])
                    self.wt_car[veh] = self.env.vehicles[veh][veh_lane]
        avg_wt_car = 0.0 if len(wait_time_car_list)<1 else sum(wait_time_car_list)/len(wait_time_car_list)
        avg_wt_firetruck = 0.0 if len(wait_time_firetruck_list)<1 else sum(wait_time_firetruck_list)/len(wait_time_firetruck_list)
        return avg_wt_car, avg_wt_firetruck

    def get_waiting_time(self):
        wait_time_per_road = []
        for p in range(self.num_green_phases):
            veh_list = self.get_veh_list(p)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_road.append(wait_time)
        return wait_time_per_road

    def get_waiting_time_all(self):
        wait_time_per_road_car = []
        wait_time_per_road_firetruck = []
        for p in range(self.num_green_phases):
            veh_list = self.get_veh_list(p)
            wait_time_car = 0.0
            wait_time_firetruck = 0.0
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                # acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                # if veh not in self.env.vehicles:
                #     self.env.vehicles[veh] = {veh_lane: acc}
                # else:
                #     self.env.vehicles[veh][veh_lane] = acc - sum(
                #         [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                if traci.vehicle.getTypeID(veh) == 'Car':
                    wait_time_car += self.env.vehicles[veh][veh_lane]
                else:
                    wait_time_firetruck += self.env.vehicles[veh][veh_lane]
            wait_time_per_road_car.append(wait_time_car)
            wait_time_per_road_firetruck.append(wait_time_firetruck)
        return wait_time_per_road_car, wait_time_per_road_firetruck

    def get_lanes_density(self):
        vehicle_size_min_gap = 7  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap))
                for lane in self.lanes]

    def get_lanes_wait_time(self):
        wt_per_lane_car = []
        wt_per_lane_firetruck = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wt_car = 0
            wt_firetruck = 0
            for veh in veh_list:
                veh_lane = self.get_edge_id(traci.vehicle.getLaneID(veh))
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])

                if traci.vehicle.getTypeID(veh) == 'Car':
                    wt_car += self.env.vehicles[veh][veh_lane]
                else:
                    wt_firetruck += self.env.vehicles[veh][veh_lane]
            wt_per_lane_car.append(wt_car)
            wt_per_lane_firetruck.append(wt_firetruck)
        return wt_per_lane_car, wt_per_lane_firetruck

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap))
                for lane in self.lanes]

    def update_veh_travelTime(self):
        for p in range(self.num_green_phases):
            veh_list = self.get_veh_list(p)
            for veh in veh_list:
                if veh not in self.veh_travel_record:
                    self.veh_travel_record[veh]= {"enter_time":traci.simulation.getCurrentTime()}
                    self.veh_travel_record[veh]["leave_time"] = traci.simulation.getCurrentTime()
                    self.veh_travel_record[veh]["type"] = traci.vehicle.getTypeID(veh)
                else:
                    self.veh_travel_record[veh]["leave_time"] = traci.simulation.getCurrentTime()

    def get_avg_travel_time(self):
        car_list = []
        firetruck_list =[]
        for veh in self.veh_travel_record:
            travel_time = self.veh_travel_record[veh]["leave_time"] - self.veh_travel_record[veh]["enter_time"]
            if self.veh_travel_record[veh]["type"] == 'Car':
                car_list.append(int(travel_time / 1000))
            else:
                firetruck_list.append(int(travel_time / 1000))
        avg_car = 0.0 if len(car_list)<1 else np.mean(car_list)
        avg_firetruck = 0.0 if len(firetruck_list)<1 else np.mean(firetruck_list)
        return avg_car, avg_firetruck


    def get_throughput(self):
        veh_list_now = []
        for p in range(self.num_green_phases):
            veh_list_now += self.get_veh_list(p)
        veh_in_num = 0
        for veh in veh_list_now:
            if veh not in self.veh_list_pre:
                veh_in_num += 1
        veh_out_num = len(self.veh_list_pre) + veh_in_num - len(veh_list_now)
        self.veh_list_pre = veh_list_now
        return veh_out_num

    @staticmethod
    def get_edge_id(lane):
        ''' Get edge Id from lane Id
        :param lane: id of the lane
        :return: the edge id of the lane
        '''
        return lane[:-2]

    def get_veh_list(self, p):
        veh_list = []
        for lane in self.edges[p]:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @DeprecationWarning
    def keep(self):
        if self.time_on_phase >= self.max_green:
            self.change()
        else:
            self.time_on_phase += self.delta_time
            traci.trafficlight.setPhaseDuration(self.id, self.delta_time)

    @DeprecationWarning
    def change(self):
        if self.time_on_phase < self.min_green:  # min green time => do not change
            self.keep()
        else:
            self.time_on_phase = self.delta_time
            traci.trafficlight.setPhaseDuration(self.id, 0)