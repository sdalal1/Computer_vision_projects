#include <gtkmm.h>

class ImageViewer : public Gtk::Window {
public:
    ImageViewer() {
        set_title("Image Viewer");
        set_border_width(10);
        set_default_size(640, 480);

        // Load the image
        Glib::RefPtr<Gdk::Pixbuf> pixbuf = Gdk::Pixbuf::create_from_file("face.bmp");
        if (pixbuf) {
            image.set(pixbuf);
            add(image);
        }

        // Connect the destroy signal
        signal_hide().connect(sigc::mem_fun(*this, &ImageViewer::on_hide));
    }

    virtual ~ImageViewer() {}

protected:
    void on_hide() {
        // Quit the GTK main loop when the window is closed
        Gtk::Main::quit();
    }

    Gtk::Image image;
};

int main(int argc, char *argv[]) {
    auto app = Gtk::Application::create(argc, argv, "org.gtkmm.example");

    ImageViewer viewer;
    viewer.show_all();

    return app->run(viewer);
}
