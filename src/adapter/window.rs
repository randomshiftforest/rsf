use std::collections::VecDeque;

pub struct Window<I: Iterator> {
    iter: I,
    buf: VecDeque<I::Item>,
    w: usize,
}

pub enum WindowUpdate<T> {
    Insert(T),
    Replace(T, T),
}

impl<I: Iterator> Window<I> {
    fn new(iter: I, w: usize) -> Self {
        assert!(w > 0, "invalid window size");
        let buf = VecDeque::with_capacity(w);
        Self { iter, buf, w }
    }
}

impl<I: Iterator> Iterator for Window<I>
where
    I::Item: Clone,
{
    type Item = WindowUpdate<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|new_item| {
            if self.buf.len() < self.w {
                self.buf.push_back(new_item.clone());
                WindowUpdate::Insert(new_item)
            } else {
                let old_item = self.buf.pop_front().unwrap();
                self.buf.push_back(new_item.clone());
                WindowUpdate::Replace(old_item, new_item)
            }
        })
    }
}

pub trait WindowIterator: Iterator + Sized {
    fn window(self, w: usize) -> Window<Self> {
        Window::new(self, w)
    }
}

impl<I: Iterator> WindowIterator for I {}
