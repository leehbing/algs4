package com.leetcode;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 26/5/2020 10:43 AM
 * @Author: lihongbing
 */
//解法二：哈希表 + 双向链表 = LinkedHashMap
//
public class LRUCache2 extends LinkedHashMap<Integer, Integer> {
    private int capacity;

    public LRUCache2(int capacity) {
        //参数true表示按照访问顺序(包括get和put)，iterator的第一个元素表示最久被访问的，最后一个元素表示最近被访问的
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity; //当容量达到上届capacity，就删除最老的，正好满足我们的需要
    }
}
