import json
import os
from collections import Counter


class Alphabet:
    def __init__(self,name,label=False,keep_growing=True,min_freq=1):
        self.__name = name
        self.UNKNOWN = "</unk>"
        self.label = label
        self.instance2index = {}
        self.instance_counter = Counter()
        self.instances = []
        self.keep_growing = keep_growing
        self.min_freq = min_freq

        if not label:
            self.default_index = 0
            self.next_index = 1
            self.add(self.UNKNOWN)
        else:
            self.next_index = 1
            self.instance2index = {'O':0}
            self.instances = ['O']
    def clear(self,keep_growing=True):
        self.instance2index = {}
        self.instances = []
        self.keep_growing = keep_growing

        self.default_index = 0
        self.next_index = 1

    def add(self, instance):
        if instance not in self.instance2index:
            self.instances.append(instance)
            self.instance2index[instance] = self.next_index
            self.next_index += 1

    def get_index(self, instance):
        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                index = self.next_index
                self.add(instance)
                return index
            else:
                return self.instance2index[self.UNKNOWN]

    def get_instance(self, index):
        if index == 0:
            return None
        try:
            return self.instances[index - 1]
        except IndexError:
            print('WARNING:Alphabet get_instance ,unknown instance index {}, return the first label.'.format(index))
            return self.instances[0]
    def get_id2instance(self):
        return {v:k for k,v in self.instance2index.items()}
    def size(self):
        return len(self.instances) + 1

    def iteritems(self):
        return self.instance2index.items()

    def enumerate_items(self, start=1):
        if start < 1 or start >= self.size():
            raise IndexError("Enumerate is allowed between [1 : size of the alphabet]")
        return zip(range(start, len(self.instances) + 1), self.instances[start - 1:])

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {"instance2index":self.instance2index,"instances":self.instances}

    def from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        saving_name = name if name else self.__name
        try:
            with open(os.path.join(output_directory, saving_name+".json"),"w") as f:
                json.dump(self.get_content(), f,ensure_ascii=False)
        except Exception as e:
            print("Exception: Alphabet is not saved: "% repr(e))

    def load(self, input_directory, name=None):
        loading_name = name if name else self.__name
        with open(os.path.join(input_directory, loading_name + ".json")) as f:
            self.from_json(json.load(f))



