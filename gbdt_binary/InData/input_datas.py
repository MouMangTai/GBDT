# -*- coding:utf-8 -*-


class IN_DataSet:
    """
    分类问题默认标签列名称为label，二元分类标签∈{-1, +1}
    回归问题也统一使用label
    """
    def __init__(self, dele, np_dataset,head_dataset):  # just for csv data format
        line_cnt = 0
        self.instances = dict()
        self.distinct_valueset = dict()  # just for real value type
        self.label = []
        self.set_feature_label(head_dataset)
        # print(np_dataset.shape)
        # num_rows = np_dataset.shape[0]
        for line in np_dataset:
            if line_cnt > 0:
                
                fields = line[dele:]
                # print(fields)
                # print(len(fields))
                # print(len(self.field_names))
                if len(fields) != len(self.field_names):
                    print("wrong fields:", line)
                    raise ValueError("fields number is wrong!")
                if line_cnt == 1:  # determine the value type
                    self.field_type = dict()
                    for i in range(0, len(self.field_names)):
                        valueSet = set()
                        try:
                            float(fields[i])
                            self.distinct_valueset[self.field_names[i]] = set()
                        except ValueError:
                       
                            valueSet.add(fields[i])
                        self.field_type[self.field_names[i]] = valueSet
                 
                if fields[len(fields) - 1] == ' <=50K' or fields[-1] == ' <=50K.':
                    # print(fields[-1]," ","ss")
                    fields[-1] = '-1'
                    self.label.append(0)
                elif fields[-1] == ' >50K' or fields[-1] == ' >50K.':
                    # print(fields[-1]," ","tt")
                    fields[-1] = '1'
                    self.label.append(1)
               
                self.instances[line_cnt] = self._construct_instance(fields)
           

            line_cnt += 1
        self.dsize = line_cnt - 1
        # print(len(self.label))
        
    def set_feature_label(self, np_dataset):
        line_cnt = 0
        self.instances = dict()
        self.distinct_valueset = dict()  # just for real value type
        self.label = []
        # print(np_dataset.shape)
        # num_rows = np_dataset.shape[0]
        for line in np_dataset:
            # print(line)
            """
            if line == "\n":
                continue
            """
          
            fields = line
         
            if line_cnt == 0:  # csv head
                lst = [i for i in range(len(fields) - 1)]
                lst2 = lst + ['label']
                self.field_names = tuple(lst2)
        
    def _construct_instance(self, fields):
        """构建一个新的样本"""
        instance = dict()
        for i in range(0, len(fields)):
            field_name = self.field_names[i]
            # print(field_name)
            real_type_mark = self.is_real_type_field(field_name)
            # real_type_mark = 0

            if real_type_mark:
                try:
                    instance[field_name] = float(fields[i])
                    self.distinct_valueset[field_name].add(float(fields[i]))
                except ValueError:
                    raise ValueError("the value is not float,"
                                     "conflict the value type at first detected")
            else:
                instance[field_name] = fields[i]
                self.field_type[field_name].add(fields[i])

        return instance

    def describe(self):
        info = "features:"+str(self.field_names)+"\n"
        info = info+"\n dataset size="+str(self.size())+"\n"
        for field in self.field_names:
            info = info+"description for field:"+field
            valueset = self.get_distinct_valueset(field)
            if self.is_real_type_field(field):
                info = info+" real value, distinct values number:"+str(len(valueset))
                info = info+" range is ["+str(min(valueset))+","+str(max(valueset))+"]\n"
            else:
                info = info+" enum type, distinct values number:"+str(len(valueset))
                info = info+" valueset="+str(valueset)+"\n"
            info = info+"#"*60+"\n"
        print(info)

    def get_instances_idset(self):
        """获取样本的id集合"""
        # print(set(self.instances.keys()))
        return set(self.instances.keys())

    def is_real_type_field(self, name):
        """判断特征类型是否是real type"""
        # print(self.field_names)
        # print(name)
        if name not in self.field_names:

             raise ValueError(" field name not in the dictionary of dataset")
        # print(self.field_type['age'])
        return len(self.field_type[name]) == 0

    def get_label_size(self, name="label"):
        if name not in self.field_names:
            raise ValueError(" there is no class label field!")
        # 因为训练样本的label列的值可能不仅仅是字符类型，也可能是数字类型
        # 如果是数字类型则field_type[name]为空
        # print(self.field_type[name])
        return len(self.field_type[name]) or len(self.distinct_valueset[name])

    def get_label_valueset(self, name="label"):
        """返回具体分离label"""
        if name not in self.field_names:
            raise ValueError(" there is no class label field!")
        return self.field_type[name] if self.field_type[name] else self.distinct_valueset[name]

    def size(self):
        """返回样本个数"""
        return len(self.instances)

    def get_instance(self, Id):
        """根据ID获取样本"""
        if Id not in self.instances:
            raise ValueError("Id not in the instances dict of dataset")
        return self.instances[Id]

    def get_attributes(self):
        """返回所有features的名称"""
        field_names = [x for x in self.field_names if x != "label"]
        return tuple(field_names)

    def get_distinct_valueset(self, name):
        if name not in self.field_names:
            raise ValueError("the field name not in the dataset field dictionary")
        if self.is_real_type_field(name):
            return self.distinct_valueset[name]
        else:
            return self.field_type[name]



