package com.leetcode;

import java.util.*;

/**
 * Created by IntelliJ IDEA
 *
 * @Date: 3/6/2020 5:14 PM
 * @Author: lihongbing
 */
//链表相关的
public class LinkedListRelated {
    //206. 反转链表
    public static ListNode reverseList(ListNode head) {
        //方法一：增加两个指针，遍历的时候保留临时指针
        //时间复杂度：O(n)，假设 n 是列表的长度，时间复杂度是 O(n)。
        //空间复杂度：O(1)
        ListNode pre = null;
        ListNode next = head;
        while (next != null) {
            ListNode temp = next.next; //必须
            next.next = pre;
            pre = next;
            next = temp;
        }
        return pre;

    }

    //206. 反转链表
    public ListNode reverseList2(ListNode head) {
        //方法二：递归思想：https://leetcode-cn.com/problems/reverse-linked-list/solution/fan-zhuan-lian-biao-by-leetcode/
        //递归比较难以理解，看评论：看了半个小时可算是把这个递归看懂了！
        // 不妨假设链表为1，2，3，4，5。按照递归，当执行reverseList（5）的时候返回了5这个节点，reverseList(4)中的p就是5这个节点，
        // 我们看看reverseList（4）接下来执行完之后，5->next = 4, 4->next = null。这时候返回了p这个节点，也就是链表5->4->null，
        // 接下来执行reverseList（3），代码解析为4->next = 3,3->next = null，这个时候p就变成了，5->4->3->null, reverseList(2),
        // reverseList(1)依次类推，p就是:5->4->3->2->1->null
        //时间复杂度：O(n)，假设 nn 是列表的长度，那么时间复杂度为 O(n)。
        //空间复杂度：O(n)，由于使用递归，将会使用隐式栈空间。递归深度可能会达到 n 层。
        if (head == null || head.next == null) return head;
        ListNode p = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }

    //92. 反转链表 II      反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
    public static ListNode reverseBetween(ListNode head, int m, int n) {
        //思想：计数器，将要反转的部分截取出来反转，然后在拼回去
        //比如输入 1 -> 2 -> 3 -> 4 -> 5 -> null     2,4
        //        n1   p         q    n2
        //首先扫描一遍链表，利用p,q记录m和n对应的位置，n1和n2记录m-1和n+1的位置
        //然后将q.next=null，针对p这个子链表进行反转，最后再拼接回去
        //时间复杂度：第一次会扫描一遍整个链表，然后会扫描一遍子链表[m,n] ，O(l+n-m),其实达不到题目所要求的只扫描一遍的要求。
        int i = 0;
        ListNode cur = head;
        ListNode p = head;
        ListNode q = null;

        ListNode n1 = null;
        ListNode n2 = null;

        while (cur != null) {
            i++;
            if (i == m - 1) n1 = cur;
            if (i == m) p = cur; //肯定不为空
            if (i == n) q = cur; //肯定不为空
            if (i == n + 1) n2 = cur;
            cur = cur.next;
        }


        q.next = null;
        //将p反转，同题目206
        ListNode pre = null;
        ListNode next = p;

        while (next != null) {
            ListNode temp = next.next;
            next.next = pre;
            pre = next;
            next = temp;
        }

        if (n1 != null) {
            n1.next = pre;
        } else {
            head = pre;
        }
        p.next = n2;


        return head;
    }
    ////思路：head表示需要反转的头节点，pre表示需要反转头节点的前驱节点
    // 同时我们也需要设置一个哑节点dummy，因为m=1时，我们可以也有前驱节点，剩余部分看代码即可。
    //    //我们需要反转n-m次，我们将head的next节点移动到需要反转链表部分的首部，需要反转链表部分剩余节点依旧保持相对顺序即可
    //反转的过程中，有点像头插法。
    //    //比如1->2->3->4->5,m=1,n=5
    //    //第一次反转：1(head) 2(next) 3 4 5 反转为 2 1 3 4 5
    //    //第二次反转：2 1(head) 3(next) 4 5 反转为 3 2 1 4 5
    //    //第三次发转：3 2 1(head) 4(next) 5 反转为 4 3 2 1 5
    //    //第四次反转：4 3 2 1(head) 5(next) 反转为 5 4 3 2 1
    //    ListNode* reverseBetween(ListNode* head, int m, int n) {
    //        ListNode *dummy=new ListNode(-1);
    //        dummy->next=head;
    //        ListNode *pre=dummy;
    //        for(int i=1;i<m;++i)pre=pre->next;
    //        head=pre->next;
    //        for(int i=m;i<n;++i){
    //            ListNode *nxt=head->next;
    //            //head节点连接nxt节点之后链表部分，也就是向后移动一位
    //            head->next=nxt->next;
    //            //nxt节点移动到需要反转链表部分的首部
    //            nxt->next=pre->next;
    //            pre->next=nxt;//pre继续为需要反转头节点的前驱节点
    //        }
    //        return dummy->next;
    //    }

    //234. 回文链表    请判断一个链表是否为回文链表。要求空间复杂度达到O(1)
    //输入: 1->2->2->1
    //输出: true
    //方法一，最简单的方法是，将值复制到数组中后用双指针法
    //时间复杂度：O(n)
    //空间复杂度：O(n)
    //方法二，递归，空间复杂度还是O(n)，太复杂了，可以看官网
    //方法三，将链表的后半部分反转（修改链表结构），然后将前半部分和后半部分进行比较。比较完成后我们应该将链表恢复原样。
    // 虽然不需要恢复也能通过测试用例，因为使用该函数的人不希望链表结构被更改。
    //          空间复杂度O(1)
    public static boolean isPalindrome(ListNode head) {
        if (head == null) return true;

        // Find the end of first half and reverse second half.
        ListNode firstHalfEnd = Utils.endOfFirstHalf(head); //通过快慢指针法找到前半部分的尾节点
        ListNode secondHalfStart = reverseList(firstHalfEnd.next);

        // Check whether or not there is a palindrome.
        ListNode p1 = head;
        ListNode p2 = secondHalfStart;
        boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) result = false;
            p1 = p1.next;
            p2 = p2.next;
        }

        // Restore the list and return the result.
        firstHalfEnd.next = reverseList(secondHalfStart);
        return result;
    }

    //160. 相交链表
    //两个链表是相交的，求相交的结点
    //方法一：暴力法
    //针对headA中的每个结点，查看在headB中是否存在，当存在就返回这个结点，这就是相交的结点
    //时间复杂度：O(mn)
    public static ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA;
        ListNode p2 = null;
        while (p1 != null) {
            p2 = headB;
            while (p2 != null) {
                if (p1 == p2) {
                    return p1;
                }
                p2 = p2.next;
            }
            p1 = p1.next;
        }
        return null;
    }

    //方法二：hashMap
    //时间复杂度 : O(m+n)
    //空间复杂度 : O(m) 或 O(n)
    public static ListNode getIntersectionNode2(ListNode headA, ListNode headB) {
        ListNode p1 = headA;
        ListNode p2 = headB;
        Map<ListNode, Integer> map = new HashMap<>();
        while (p1 != null) {
            map.put(p1, 1);
            p1 = p1.next;
        }
        while (p2 != null) {
            if (map.containsKey(p2)) return p2;
            p2 = p2.next;
        }
        return null;
    }
    //方法三：双指针法，官方还有一种解法


    //21. 合并两个有序链表
    //将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0);
        ListNode p = dummyHead;
        int cur = 0;
        while (l1 != null || l2 != null) {
            if (l1 == null) {
                cur = l2.val;
                l2 = l2.next;
            } else if (l2 == null) {
                cur = l1.val;
                l1 = l1.next;
            } else if (l1.val < l2.val) {
                cur = l1.val;
                l1 = l1.next;
            } else if (l1.val >= l2.val) {
                cur = l2.val;
                l2 = l2.next;
            }
            p.next = new ListNode(cur);
            p = p.next;
        }
        return dummyHead.next;
    }

    //88. 合并两个有序数组
    //给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。
    //方法一：合并后排序
    //时间复杂度 : O((n+m)log(n+m))
    //空间复杂度 : O(1)
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        System.arraycopy(nums2, 0, nums1, m, n);
        Arrays.sort(nums1);
    }

    //方法二：双指针 / 从前往后
    //时间复杂度 : O(n + m)
    //空间复杂度 : O(m)
    public void merge2(int[] nums1, int m, int[] nums2, int n) {
        // Make a copy of nums1.
        int[] nums1_copy = new int[m];
        System.arraycopy(nums1, 0, nums1_copy, 0, m);

        // Two get pointers for nums1_copy and nums2.
        int p1 = 0;
        int p2 = 0;

        // Set pointer for nums1
        int p = 0;

        // Compare elements from nums1_copy and nums2
        // and add the smallest one into nums1.
        while ((p1 < m) && (p2 < n))
            nums1[p++] = (nums1_copy[p1] < nums2[p2]) ? nums1_copy[p1++] : nums2[p2++];

        // if there are still elements to add
        if (p1 < m)
            System.arraycopy(nums1_copy, p1, nums1, p1 + p2, m + n - p1 - p2);
        if (p2 < n)
            System.arraycopy(nums2, p2, nums1, p1 + p2, m + n - p1 - p2);
    }

    //141. 环形链表
    //给定一个链表，判断链表中是否有环。你能用 O(1)（即，常量）内存解决此问题吗？
    //
    //为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
    //方法一：利用hashmap
    //时间复杂度：O(n)
    //空间复杂度：O(n),因为要存储map（存的是ListNode的饮用）
    public boolean hasCycle(ListNode head) {
        Map<ListNode, Integer> map = new HashMap();
        ListNode p = head;
        while (p != null) {
            if (map.containsKey(p)) {
                return true;
            } else {
                map.put(p, 1);
            }
            p = p.next;

        }
        return false;

    }

    //方法二：双指针，快慢指针
    //过使用具有 不同速度 的快、慢两个指针遍历链表，空间复杂度可以被降低至 O(1)。慢指针每次移动一步，而快指针每次移动两步。
    //
    //如果列表中不存在环，最终快指针将会最先到达尾部，此时我们可以返回 false。
    //现在考虑一个环形链表，把慢指针和快指针想象成两个在环形赛道上跑步的运动员（分别称之为慢跑者与快跑者）。而快跑者最终一定会追上慢跑者。这是为什么呢？
    // 考虑下面这种情况（记作情况 A）- 假如快跑者只落后慢跑者一步，在下一次迭代中，它们就会分别跑了一步或两步并相遇。
    //
    //其他情况又会怎样呢？例如，我们没有考虑快跑者在慢跑者之后两步或三步的情况。但其实不难想到，因为在下一次或者下下次迭代后，又会变成上面提到的情况 A。
    //时间复杂度：O(n)
    //空间复杂度：O(1)
    public boolean hasCycle2(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }

    //142. 环形链表 II
    //给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
    //
    //为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
    //
    //说明：不允许修改给定的链表   ，你是否可以不用额外空间解决此题？
    //方法一：利用hashmap
    //时间复杂度：O(n)
    //空间复杂度：O(n),因为要存储map（存的是ListNode的饮用）
    public ListNode detectCycle(ListNode head) {
        Map<ListNode, Integer> map = new HashMap();
        ListNode p = head;
        while (p != null) {
            if (map.containsKey(p)) {
                return p;
            } else {
                map.put(p, 1);
            }
            p = p.next;

        }
        return null;
    }
    //方法二：快慢指针
    //快慢指针跑圈，可以看看官方怎么解答的


    //202. 快乐数
    //编写一个算法来判断一个数 n 是不是快乐数。
    //
    //「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果 可以变为  1，那么这个数就是快乐数。
    //
    //如果 n 是快乐数就返回 True ；不是，则返回 False 。
    //输入：19
    //输出：true
    //解释：
    //12 + 92 = 82
    //82 + 22 = 68
    //62 + 82 = 100
    //12 + 02 + 02 = 1
    //根据我们的探索，我们猜测会有以下三种可能。  ==》环形链表
    //最终会得到 1。
    //最终会进入循环。
    //值会越来越大，最后接近无穷大。       这种情况不可能发生
    private int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }

    public boolean isHappy(int n) {
        Set<Integer> seen = new HashSet<>();
        while (n != 1 && !seen.contains(n)) {
            seen.add(n);
            n = getNext(n);
        }
        return n == 1;
    }


    //面试题 02.01. 移除重复节点
    //编写代码，移除未排序链表中的重复节点。保留最开始出现的节点。
    //
    //示例1:
    //
    // 输入：[1, 2, 3, 3, 2, 1]
    // 输出：[1, 2, 3]
    //示例2:
    //
    // 输入：[1, 1, 1, 1, 2]
    // 输出：[1, 2]
    //提示：
    //
    //链表长度在[0, 20000]范围内。
    //链表元素在[0, 20000]范围内。
    //进阶：
    //
    //如果不得使用临时缓冲区，该怎么解决？
    //方法一：利用hashmap
    //时间复杂度：O(N)，其中 N 是给定链表中节点的数目。
    //空间复杂度：O(N)。在最坏情况下，给定链表中每个节点都不相同，哈希表中需要存储所有的 N 个值。
    public ListNode removeDuplicateNodes(ListNode head) {
        Map<Integer, Integer> values = new HashMap<>(); //改成HashSet更好
        ListNode dummyHead = new ListNode(-1);
        dummyHead.next = head;
        ListNode current = head;
        ListNode pre = dummyHead;
        while (current != null) {
            if (values.containsKey(current.val)) {
                pre.next = current.next;
                current = current.next;
            } else {
                values.put(current.val, 1);
                pre = current;
                current = current.next;
            }
        }
        return dummyHead.next;
    }

    //方法二：两重循环
    //考虑题目描述中的「进阶」部分，是否存在一种不使用临时缓冲区（也就是方法一中的哈希表）的方法呢？
    //
    //不幸的是，在保证方法一时间复杂度 O(N)的前提下，是不存在这样的方法的。
    // 因此我们需要增加时间复杂度，使得我们可以仅使用常数的空间来完成本题。一种简单的方法是，我们在给定的链表上使用两重循环，
    // 其中第一重循环从链表的头节点开始，枚举一个保留的节点，这是因为我们保留的是「最开始出现的节点」。
    // 第二重循环从枚举的保留节点开始，到链表的末尾结束，将所有与保留节点相同的节点全部移除。
    //
    //方法二的细节部分与方法一类似。第一重循环枚举保留的节点本身，而为了编码方便，第二重循环可以枚举待移除节点的前驱节点，
    // 方便我们对节点进行移除。这样一来，我们使用「时间换空间」的方法，就可以不使用临时缓冲区解决本题了。
    //时间复杂度：O(N^2)，其中 N 是给定链表中节点的数目。
    //空间复杂度：O(1)。
    public ListNode removeDuplicateNodes2(ListNode head) {
        ListNode ob = head;
        while (ob != null) {
            ListNode oc = ob;
            while (oc.next != null) {
                if (oc.next.val == ob.val) {
                    oc.next = oc.next.next;
                } else {
                    oc = oc.next;
                }
            }
            ob = ob.next;
        }
        return head;
    }


    //83. 删除排序链表中的重复元素
    //给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
    //
    //示例 1:
    //输入: 1->1->2
    //输出: 1->2
    //示例 2:
    //输入: 1->1->2->3->3
    //输出: 1->2->3

    //复杂度分析
    //时间复杂度：O(n)，因为列表中的每个结点都检查一次以确定它是否重复，所以总运行时间为 O(n)，其中 n 是列表中的结点数。
    //空间复杂度：O(1)，没有使用额外的空间。
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return head;
        ListNode p0 = head;
        ListNode p1 = head.next;
        while (p1 != null) {
            if (p0.val == p1.val) {
                p0.next = p1.next;
                p1 = p1.next;
            } else {
                p0 = p1;
                p1 = p1.next;
            }
        }

        return head;
    }

    //面试题 02.02. 返回倒数第 k 个节点
    //实现一种算法，找出单向链表中倒数第 k 个节点。返回该节点的值。
    //
    //注意：本题相对原题稍作改动
    //示例：
    //输入： 1->2->3->4->5 和 k = 2
    //输出： 4
    //1，双指针求解
    //这题要求链表的倒数第k个节点，最简单的方式就是使用两个指针，第一个指针先移动k步，然后第二个指针再从头开始，
    // 这个时候这两个指针同时移动（都只移动一个节点），当第一个指针到链表的末尾的时候，返回第二个指针即可。
    //复杂度：O(n)
    public int kthToLast(ListNode head, int k) {
        ListNode first = head;
        ListNode second = head;
        //第一个指针先走k步
        while (k-- > 0) {
            first = first.next;
        }
        //然后两个指针在同时前进
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        return second.val;
    }


    //197. 上升的温度

    //MySQL
    //编写一个 SQL 查询，来查找与之前（昨天的）日期相比温度更高的所有日期的 id 。
    //
    //返回结果 不要求顺序 。
    //查询结果格式如下例：
    //
    //Weather
    //+----+------------+-------------+
    //| id | recordDate | Temperature |
    //+----+------------+-------------+
    //| 1  | 2015-01-01 | 10          |
    //| 2  | 2015-01-02 | 25          |
    //| 3  | 2015-01-03 | 20          |
    //| 4  | 2015-01-04 | 30          |
    //+----+------------+-------------+
    //
    //Result table:
    //+----+
    //| id |
    //+----+
    //| 2  |
    //| 4  |
    //+----+
    //2015-01-02 的温度比前一天高（10 -> 25）
    //2015-01-04 的温度比前一天高（20 -> 30）


//    SELECT
//    weather.id AS 'Id'
//    FROM
//            weather
//    JOIN
//    weather w ON DATEDIFF(weather.date, w.date) = 1
//    AND weather.Temperature > w.Temperature
//    ;


}
