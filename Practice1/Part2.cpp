#include <iostream>      // Для вывода в консоль
#include <omp.h>         // Для OpenMP (используется в комментариях)

using namespace std;     // Чтобы не писать std::

// Структура узла — используется во всех динамических структурах
struct Node {
    int data;            // Значение узла
    Node* next;          // Указатель на следующий узел
    Node(int val) : data(val), next(nullptr) {}  // Конструктор
};

// Односвязный список
class SinglyLinkedList {
private:
    Node* head;          // Указатель на голову списка
public:
    SinglyLinkedList() : head(nullptr) {}  // Конструктор — пустой список

    // Добавление элемента в конец
    void push_back(int val) {
        Node* new_node = new Node(val);
        if (!head) {
            head = new_node;
            return;
        }
        Node* temp = head;
        while (temp->next) temp = temp->next;
        temp->next = new_node;
    }

    // Удаление первого вхождения значения
    void remove(int val) {
        if (!head) return;
        if (head->data == val) {
            Node* temp = head;
            head = head->next;
            delete temp;
            return;
        }
        Node* temp = head;
        while (temp->next && temp->next->data != val) temp = temp->next;
        if (temp->next) {
            Node* to_delete = temp->next;
            temp->next = to_delete->next;
            delete to_delete;
        }
    }

    // Поиск элемента
    bool find(int val) {
        Node* temp = head;
        while (temp) {
            if (temp->data == val) return true;
            temp = temp->next;
        }
        return false;
    }

    // Вывод списка
    void print() {
        Node* temp = head;
        while (temp) {
            cout << temp->data << " -> ";
            temp = temp->next;
        }
        cout << "nullptr\n";
    }

    // Деструктор — освобождение памяти
    ~SinglyLinkedList() {
        while (head) {
            Node* temp = head;
            head = head->next;
            delete temp;
        }
    }
};

// Стек (LIFO)
class Stack {
private:
    Node* top;           // Указатель на вершину
public:
    Stack() : top(nullptr) {}

    void push(int val) {
        Node* new_node = new Node(val);
        new_node->next = top;
        top = new_node;
    }

    int pop() {
        if (!top) {
            cout << "Stack underflow!\n";
            return -1;
        }
        Node* temp = top;
        int val = temp->data;
        top = top->next;
        delete temp;
        return val;
    }

    bool isEmpty() { return top == nullptr; }

    ~Stack() {
        while (top) {
            Node* temp = top;
            top = top->next;
            delete temp;
        }
    }
};

// Очередь (FIFO)
class Queue {
private:
    Node* front;         // Указатель на начало
    Node* rear;          // Указатель на конец
public:
    Queue() : front(nullptr), rear(nullptr) {}

    void enqueue(int val) {
        Node* new_node = new Node(val);
        if (!rear) {
            front = rear = new_node;
            return;
        }
        rear->next = new_node;
        rear = new_node;
    }

    int dequeue() {
        if (!front) {
            cout << "Queue underflow!\n";
            return -1;
        }
        Node* temp = front;
        int val = temp->data;
        front = front->next;
        if (!front) rear = nullptr;
        delete temp;
        return val;
    }

    bool isEmpty() { return front == nullptr; }

    ~Queue() {
        while (front) {
            Node* temp = front;
            front = front->next;
            delete temp;
        }
    }
};

int main() {
    cout << "Part 2: Data structures\n\n";

    // Демонстрация односвязного списка
    SinglyLinkedList list;
    list.push_back(10);
    list.push_back(20);
    list.push_back(30);
    cout << "Singly linked list: ";
    list.print();

    list.remove(20);
    cout << "After removing 20: ";
    list.print();

    cout << "Search for 30: " << (list.find(30) ? "Found" : "Not found") << endl << endl;

    // Демонстрация стека
    Stack stack;
    stack.push(1);
    stack.push(2);
    stack.push(3);
    cout << "Stack pop: " << stack.pop() << " (expected 3)\n";
    cout << "Is stack empty? " << (stack.isEmpty() ? "Yes" : "No") << endl << endl;

    // Демонстрация очереди
    Queue queue;
    queue.enqueue(100);
    queue.enqueue(200);
    queue.enqueue(300);
    cout << "Queue dequeue: " << queue.dequeue() << " (expected 100)\n";
    cout << "Is queue empty? " << (queue.isEmpty() ? "Yes" : "No") << endl;

    // Примечание: для параллельного добавления/удаления нужна синхронизация (#pragma omp critical)
    return 0;
}