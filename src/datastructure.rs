pub struct ParkingSystem {
    capacity: Vec<i32>,
}

impl ParkingSystem {
    pub fn new(big: i32, medium: i32, small: i32) -> Self {
        Self {
            capacity: vec![big, medium, small],
        }
    }

    pub fn add_car(&mut self, car_type: i32) -> bool {
        let n = car_type as usize - 1;
        if self.capacity[n] > 0 {
            self.capacity[n] -= 1;
            true
        } else {
            false
        }
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }

    pub fn to_link(nums: Vec<i32>) -> Option<Box<ListNode>> {
      let mut nums = nums.clone();
      let mut node = None;
      nums.reverse();
      for num in nums{
        node = Some(Box::new(ListNode{next: node, val: num}));
      }
      node
  }
}
