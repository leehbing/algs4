package com.leetcode;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 26/5/2020 9:40 AM
 * @Author: lihongbing
 */

//460. LFU缓存   Least Frequently Used ，最近最少使用算法
// 如果一个数据在最近一段时间很少被访问到，那么可以认为在将来它被访问的可能性也很小。因此，当空间满时，最小频率访问的数据最先被淘汰
//写法2：HashMap<Integer, Node> cache 存缓存的内容;
//      min 是最小访问频次;
//      HashMap<Integer, DoublyLinkedList>freqMap 存每个访问频次对应的 Node 的双向链表
// （写法 2 与写法 1 一样，只不过将 JDK 自带的 LinkedHashSet 双向链表实现改成了自定义的双向链表 DoublyLinkedList，减少了一些哈希相关的耗时）
//
//
//时间复杂度：O(1)
//
public class LFUCache2 {
    Map<Integer, Node> cache; // 存储缓存的内容
    Map<Integer, DoublyLinkedList> freqMap; // 存储每个频次对应的双向链表
    int size;
    int capacity;
    int min; // 存储当前最小频次

    public LFUCache2(int capacity) {
        cache = new HashMap<>(capacity);
        freqMap = new HashMap<>();
        this.capacity = capacity;
    }

    public int get(int key) {
        Node node = cache.get(key);
        if (node == null) {
            return -1;
        }
        freqInc(node);
        return node.value;
    }

    public void put(int key, int value) {
        if (capacity == 0) {
            return;
        }
        Node node = cache.get(key);
        if (node != null) {
            node.value = value;
            freqInc(node);
        } else {
            if (size == capacity) {
                DoublyLinkedList minFreqLinkedList = freqMap.get(min);
                cache.remove(minFreqLinkedList.tail.pre.key);
                minFreqLinkedList.removeNode(minFreqLinkedList.tail.pre); // 这里不需要维护min, 因为下面add了newNode后min肯定是1.
                size--;
            }
            Node newNode = new Node(key, value);
            cache.put(key, newNode);
            DoublyLinkedList linkedList = freqMap.get(1);
            if (linkedList == null) {
                linkedList = new DoublyLinkedList();
                freqMap.put(1, linkedList);
            }
            linkedList.addNode(newNode);
            size++;
            min = 1;
        }
    }

    void freqInc(Node node) {
        // 从原freq对应的链表里移除, 并更新min
        int freq = node.freq;
        DoublyLinkedList linkedList = freqMap.get(freq);
        linkedList.removeNode(node);
        if (freq == min && linkedList.head.post == linkedList.tail) {
            min = freq + 1;
        }
        // 加入新freq对应的链表
        node.freq++;
        linkedList = freqMap.get(freq + 1);
        if (linkedList == null) {
            linkedList = new DoublyLinkedList();
            freqMap.put(freq + 1, linkedList);
        }
        linkedList.addNode(node);
    }


    class Node {
        int key;
        int value;
        int freq = 1;
        Node pre;
        Node post;

        public Node() {
        }

        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }


    class DoublyLinkedList {
        Node head;
        Node tail;

        public DoublyLinkedList() {
            head = new Node();
            tail = new Node();
            head.post = tail;
            tail.pre = head;
        }

        void removeNode(Node node) { //删除指定的结点，因为不需要查找，所以是O(1)
            node.pre.post = node.post;
            node.post.pre = node.pre;
        }

        void addNode(Node node) {//头插法
            node.post = head.post;
            head.post.pre = node;
            head.post = node;
            node.pre = head;
        }
    }
}






