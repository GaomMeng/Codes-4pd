# Definition for singly-linked list.
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        dummy = ListNode(-1, None)

        def add_node(i):
            if not lists[i]:
                print(f'list {i} is None')
                return None
            tmp = dummy
            while tmp:
                if not tmp.next:
                    tmp.next = ListNode(i, None)
                    break
                if lists[i].val < lists[tmp.next.val].val:
                    tmp.next = ListNode(i, tmp.next)
                    break
                tmp = tmp.next

        def pl(node):
            while node:
                print(f'{node.val} -> ', end='')
                node = node.next
            print('None')

        pl(dummy)
        for i in range(len(lists)):
            add_node(i)
        pl(dummy)

        nd = ListNode(float('-Inf'), None)
        tmp = nd
        ii = 0
        while dummy.next and ii < 10:
            ii += 1
            print(f'\ni={ii}', [v.val if v else None for v in lists], end='   ')
            pl(dummy)
            pl(nd)


            node, i = lists[dummy.next.val], dummy.next.val
            lists[i] = node.next

            node.next = None
            tmp.next = node
            tmp = tmp.next

            dummy.next = dummy.next.next
            pl(dummy)
            add_node(i)
        
        return nd.next
    


# Example usage:
l1 = ListNode(1, ListNode(2, ListNode(4)))
l2 = ListNode(1, ListNode(3, ListNode(4)))
l3 = ListNode(2, ListNode(6))
lists = [l1, l2, l3]
s = Solution()
merged_list = s.mergeKLists(lists)
while merged_list:
    print(merged_list.val)
    merged_list = merged_list.next

# [[1, 2, 4], [1, 3, 4], [2, 6]]